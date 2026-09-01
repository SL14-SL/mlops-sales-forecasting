import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.responses import JSONResponse

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schema import PredictionRequest, PredictionResponse
from src.configs.loader import load_config, get_path

from src.monitoring.data_quality import initialize_data_quality_reference_cache, build_reference_category_cache
from src.monitoring.config import get_serving_settings, get_data_quality_settings
from src.monitoring.serving import get_summary, set_serving_readiness




from src.api import serving_state
from src.api.dependencies import get_api_key
from src.api.middleware import serving_monitoring_middleware
from src.api.serving_state import (
    reload_complete_serving_bundle,
)
from src.api.prediction_handler import handle_prediction
from src.api.routers.health import livez, readyz, health
from src.api.routers.admin import (
    reload_model, 
    reload_feature_state, 
    reload_serving_state,
    list_serving_releases,
    rollback_serving_release,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


# 1. Load configuration and paths
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]
VALIDATED_PATH = get_path("validated_data")
MODELS_PATH = get_path("models")
FEATURES_PATH = get_path("features")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", CFG.get("gcp", {}).get("gcs", {}).get("bucket_name"))

# Global variables for caching
model = None
store_metadata = None
store_state = None
model_type = "xgboost"
target_transformation = "none"
serving_alias = "unknown"
model_uri = None
dq_reference_categories: dict[str, set[str]] = {}
serving_model_version = None
serving_model_run_id = None
known_calendar = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize and clean up the forecasting API.

    Startup behavior:
    - smoke-test mode skips external artifact loading
    - the complete serving bundle is loaded and validated
    - an incomplete bundle is never activated
    - startup continues in degraded mode if loading fails

    Readiness reports whether a complete bundle is active.
    """
    global dq_reference_categories

    if os.getenv("SMOKE_TEST") == "1":
        logger.info(
            "Smoke test mode enabled. "
            "Skipping serving bundle startup loading."
        )

        try:
            yield
        finally:
            logger.info(
                "Shutdown: cleaning up API assets."
            )

        return

    try:
        # ---------------------------------------------
        # 1. Load and atomically activate serving bundle
        # ---------------------------------------------
        try:
            reload_result = (
                reload_complete_serving_bundle()
            )

            logger.info(
                "Initial serving bundle loaded | "
                "alias=%s | version=%s | run_id=%s",
                reload_result["serving_alias"],
                reload_result["model_version"],
                reload_result["model_run_id"],
            )

        except Exception as serving_error:
            logger.exception(
                "Initial serving bundle load failed. "
                "API will start in degraded mode: %s",
                serving_error,
            )

        # ---------------------------------------------
        # 2. Initialize data-quality reference cache
        # ---------------------------------------------
        try:
            reference_df = (
                initialize_data_quality_reference_cache()
            )

            dq_reference_categories = (
                build_reference_category_cache(
                    reference_df,
                    categorical_reference_features=(
                        get_data_quality_settings().get(
                            "categorical_reference_features",
                            [],
                        )
                    ),
                )
            )

            logger.info(
                "Data-quality reference cache initialized."
            )

        except Exception as dq_error:
            logger.warning(
                "Data-quality reference cache initialization "
                "failed: %s. Continuing with an empty cache.",
                dq_error,
            )
            dq_reference_categories = {}

        logger.info(
            "Startup sequence finished. API listening."
        )

        # The application runs between yield and finally.
        yield

    finally:
        logger.info(
            "Shutdown: cleaning up API assets."
        )

app = FastAPI(title="Blueprint Sales Forecasting API", lifespan=lifespan)

SERVING_CFG = get_serving_settings()


app.middleware("http")(
    serving_monitoring_middleware
)

if SERVING_CFG.get("metrics_endpoint_enabled", True):
    @app.get("/metrics", include_in_schema=False)
    def metrics():
        set_serving_readiness(
            serving_state
            .active_serving_bundle
            is not None
        )

        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

if SERVING_CFG.get("summary_endpoint_enabled", True):
    @app.get("/monitoring/summary", include_in_schema=False)
    def monitoring_summary():
        """
        Return the current in-memory serving-monitoring summary.
        """
        window_seconds = SERVING_CFG.get("summary_window_seconds", 900)
        return JSONResponse(get_summary(window_seconds=window_seconds))
    
app.get("/livez")(
    livez
)

app.get("/readyz")(
    readyz
)

app.post("/admin/reload-model")(
    reload_model
)

app.post("/admin/reload-serving-state")(
    reload_serving_state
)

app.post("/admin/reload-feature-state")(
    reload_feature_state
)

app.get("/health")(
    health
)

app.get("/admin/serving-releases")(
    list_serving_releases
)

app.post("/admin/rollback-serving-release")(
    rollback_serving_release
)

MAX_BATCH_ROWS = 5000


@app.post(
    "/predict",
    dependencies=[
        Depends(get_api_key)
    ],
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
):
    """
    Generate forecasts for the supplied store and date combinations.

    The endpoint uses the currently active immutable serving bundle and records
    request, data-quality and production-prediction monitoring information.

    Returns:
        Forecasts together with serving-release lineage and request metadata.

    Raises:
        HTTPException: If the request is invalid, no serving bundle is ready or
            prediction execution fails.
    """
    request_bundle = (
        serving_state
        .active_serving_bundle
    )

    if request_bundle is None:
        logger.error(
            "Predict called without an "
            "active serving bundle."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "No complete serving "
                "bundle is active."
            ),
        )

    if (
        len(payload.inputs)
        > MAX_BATCH_ROWS
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                "Batch too large. "
                "Max supported rows: "
                f"{MAX_BATCH_ROWS}"
            ),
        )

    try:
        return handle_prediction(
            payload=payload,
            bundle=request_bundle,
            reference_categories=(
                dq_reference_categories
            ),
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Prediction failed",
            extra={
                "path": "/predict",
                "release_id": (
                    request_bundle
                    .release_id
                ),
            },
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction failed: "
                f"{error}"
            ),
        ) from error