import os
import traceback
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import mlflow
from fastapi import FastAPI, HTTPException, Security, Depends, Response, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse, JSONResponse

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.status import HTTP_403_FORBIDDEN

from src.api.schema import PredictionRequest, PredictionResponse, ServingRollbackRequest
from src.configs.loader import load_config, get_path

from src.monitoring.prediction_logger import log_prediction
from src.monitoring.data_quality import initialize_data_quality_reference_cache, build_reference_category_cache, log_data_quality_runtime
from src.monitoring.config import get_serving_settings, get_data_quality_settings
from src.monitoring.serving import normalize_path, observe_request, get_summary, should_ignore_path

from src.training.target_transform import inverse_transform_target
from src.inference.pipeline import (
    apply_prediction_postprocessing,
    validate_prediction_input,
    align_features_for_model,
)
from src.inference.adapters import (
    request_to_dataframe,
    resolve_forecasting_store_id,
    resolve_open_flags,
)
from src.inference.model_manager import (
    reload_serving_model as reload_model_state,
    load_store_state,
    load_serving_bundle_for_release
)
from src.inference.model_manager import (
    load_serving_bundle,
)

from src.inference.serving_release import activate_release_pointer, list_serving_release_manifests, load_active_release_id

from src.inference.serving_bundle import ServingBundle

from src.data.features.build_features import preprocess_data

from src.inference.forecasting_policy import (
    finalize_forecasting_feature_frame,
    inject_forecasting_state_features,
    merge_request_with_calendar,
    merge_request_with_metadata,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)

def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) *1000, 2)

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
active_serving_bundle: ServingBundle | None = None

# Define the header name for the API Key
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )

def reload_serving_model() -> dict:
    """
    Reload model state and update API globals.
    """
    global model, model_type, target_transformation, serving_alias, model_uri
    global serving_model_version, serving_model_run_id

    state = reload_model_state(
        model_name=MODEL_NAME,
        cfg=CFG,
    )

    model = state["model"]
    model_type = state["model_type"]
    target_transformation = state["target_transformation"]
    serving_alias = state["serving_alias"]
    model_uri = state["model_uri"]
    serving_model_version = state["serving_model_version"]
    serving_model_run_id = state["serving_model_run_id"]

    return {
        "model_name": MODEL_NAME,
        "serving_alias": serving_alias,
        "model_version": serving_model_version,
        "model_run_id": serving_model_run_id,
        "model_uri": model_uri,
        "target_transformation": target_transformation,
    }

def activate_serving_bundle(
    bundle: ServingBundle,
) -> dict:
    """
    Atomically replace the active in-memory serving state.
    """
    global active_serving_bundle
    global model, model_type, target_transformation
    global serving_alias, model_uri
    global serving_model_version, serving_model_run_id
    global store_metadata, store_state, known_calendar

    # One authoritative snapshot reference.
    active_serving_bundle = bundle

    # Compatibility assignments for existing inference code.
    model = bundle.model
    model_type = bundle.model_type
    target_transformation = bundle.target_transformation
    serving_alias = bundle.serving_alias
    model_uri = bundle.model_uri
    serving_model_version = bundle.model_version
    serving_model_run_id = bundle.model_run_id
    store_metadata = bundle.store_metadata
    store_state = bundle.store_state
    known_calendar = bundle.known_calendar

    return {
        "release_id": bundle.release_id,
        "model_name": bundle.model_name,
        "serving_alias": bundle.serving_alias,
        "model_version": bundle.model_version,
        "model_run_id": bundle.model_run_id,
        "model_uri": bundle.model_uri,
        "target_transformation": bundle.target_transformation,
        "store_metadata_loaded": True,
        "state_loaded": True,
        "calendar_loaded": True,
    }

def reload_complete_serving_bundle() -> dict:
    """
    Load and validate a candidate bundle before activating it.
    """
    candidate_bundle = load_serving_bundle(
        model_name=MODEL_NAME,
        cfg=CFG,
        validated_path=VALIDATED_PATH,
        features_path=FEATURES_PATH,
        models_path=MODELS_PATH,
        gcs_bucket=GCS_BUCKET,
    )

    return activate_serving_bundle(candidate_bundle)


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


@app.middleware("http")
async def serving_monitoring_middleware(request: Request, call_next):
    if not SERVING_CFG.get("enabled", True):
        return await call_next(request)

    raw_path = request.url.path

    if should_ignore_path(raw_path, SERVING_CFG.get("ignored_paths")):
        return await call_next(request)

    method = request.method
    path = normalize_path(raw_path, SERVING_CFG.get("track_paths"))
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        latency_seconds = time.perf_counter() - start
        observe_request(
            method=method,
            path=path,
            status_code=status_code,
            latency_seconds=latency_seconds,
        )


if SERVING_CFG.get("metrics_endpoint_enabled", True):
    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return PlainTextResponse(
            generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

if SERVING_CFG.get("summary_endpoint_enabled", True):
    @app.get("/monitoring/summary", include_in_schema=False)
    def monitoring_summary():
        window_seconds = SERVING_CFG.get("summary_window_seconds", 900)
        return JSONResponse(get_summary(window_seconds=window_seconds))
    
@app.get("/livez")
def livez():
    """
    Report whether the API process is running.

    Liveness does not require a loaded serving bundle.
    """
    return {
        "status": "alive",
        "service": CFG.get(
            "project_name",
            "sales-forecasting-api",
        ),
        "environment": CFG.get(
            "environment",
            "unknown",
        ),
    }


@app.get("/readyz")
def readyz():
    """
    Report whether the API can safely serve predictions.
    """
    if active_serving_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="No complete serving bundle is active.",
        )

    bundle = active_serving_bundle

    return {
        "status": "ready",
        "serving_bundle_loaded": True,
        "model_name": bundle.model_name,
        "model_type": bundle.model_type,
        "target_transformation": (
            bundle.target_transformation
        ),
        "serving_alias": bundle.serving_alias,
        "model_version": bundle.model_version,
        "model_run_id": bundle.model_run_id,
        "model_uri": bundle.model_uri,
        "store_metadata_loaded": (
            bundle.store_metadata is not None
        ),
        "state_loaded": (
            bundle.store_state is not None
        ),
        "calendar_loaded": (
            bundle.known_calendar is not None
        ),
        "release_id": bundle.release_id,
    }

@app.post("/admin/reload-model")
def reload_model(api_key: str = Depends(get_api_key)):
    """
    Reload the current champion forecasting model from MLflow.

    Used after a new champion model version has been promoted.
    """
    try:
        result = reload_serving_model()

    except Exception as error:
        logger.error("Model reload failed: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(error)}",
        )

    return {
        "status": "reloaded",
        **result,
    }

@app.post("/admin/reload-serving-state")
def reload_serving_state(
    api_key: str = Depends(get_api_key),
):
    """
    Atomically reload the complete forecasting serving bundle.
    """
    try:
        result = reload_complete_serving_bundle()

    except Exception as error:
        logger.error(
            "Serving bundle reload failed: %s",
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Serving bundle reload failed. "
                "The previous serving state remains active. "
                f"Reason: {error}"
            ),
        ) from error

    return {
        "status": "reloaded",
        **result,
    }

@app.post("/admin/reload-feature-state")
def reload_feature_state(api_key: str = Depends(get_api_key)):
    """
    Reload only the forecasting feature state.

    This endpoint is used after newly available ground truth has been
    appended to latest_state.json. It does not reload or replace the model.
    """
    global store_state

    try:
        updated_state = load_store_state(
            models_path=MODELS_PATH,
            gcs_bucket=GCS_BUCKET,
        )

        if updated_state is None:
            updated_state = {}

        store_state = updated_state

    except Exception as error:
        logger.error(
            "Feature state reload failed: %s",
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Feature state reload failed: {str(error)}",
        )

    return {
        "status": "reloaded",
        "state_loaded": store_state is not None,
        "state_entities": len(store_state or {}),
    }

@app.get("/health")
def health(response: Response):
    is_healthy = (
        model is not None
        and store_metadata is not None
        and store_state is not None
    )

    if not is_healthy:
        response.status_code = 503

    return {
        "status": "online" if is_healthy else "degraded",
        "model_loaded": model is not None,
        "store_metadata_loaded": store_metadata is not None,
        "state_loaded": store_state is not None,
        "calendar_loaded": known_calendar is not None,
        "model_type": model_type,
        "target_transformation": target_transformation,
        "model_name": MODEL_NAME,
        "tracking_uri": mlflow.get_tracking_uri(),
        "serving_alias": serving_alias,
        "model_uri": model_uri,
        "model_version": serving_model_version,
        "model_run_id": serving_model_run_id,
        "release_id": (
            active_serving_bundle.release_id
            if active_serving_bundle is not None
            else None
        ),

    }

@app.get("/admin/serving-releases")
def list_serving_releases(
    api_key: str = Depends(get_api_key),
):
    active_release_id = (
        load_active_release_id(
            models_path=MODELS_PATH,
        )
    )

    manifests = (
        list_serving_release_manifests(
            models_path=MODELS_PATH,
        )
    )

    return {
        "active_release_id": (
            active_release_id
        ),
        "releases": [
            {
                "release_id": (
                    manifest.release_id
                ),
                "active": (
                    manifest.release_id
                    == active_release_id
                ),
                "created_at_utc": (
                    manifest.created_at_utc
                ),
                "model_name": (
                    manifest.model_name
                ),
                "model_version": (
                    manifest.model_version
                ),
                "model_run_id": (
                    manifest.model_run_id
                ),
                "dataset_version": (
                    manifest.dataset_version
                ),
                "git_commit": (
                    manifest.git_commit
                ),
            }
            for manifest in manifests
        ],
    }

@app.post("/admin/rollback-serving-release")
def rollback_serving_release(
    payload: ServingRollbackRequest,
    api_key: str = Depends(get_api_key),
):
    """
    Validate and atomically activate a previously published release.
    """
    previous_release_id = (
        load_active_release_id(
            models_path=MODELS_PATH,
        )
    )

    if payload.release_id == previous_release_id:
        return {
            "status": "unchanged",
            "release_id": previous_release_id,
            "previous_release_id": (
                previous_release_id
            ),
        }

    pointer_changed = False

    try:
        # Fully load model and artifacts before changing the pointer.
        candidate_bundle = (
            load_serving_bundle_for_release(
                release_id=payload.release_id,
                model_name=MODEL_NAME,
                cfg=CFG,
                models_path=MODELS_PATH,
            )
        )

        # The target bundle is valid. Persist the new active release.
        activate_release_pointer(
            models_path=MODELS_PATH,
            release_id=payload.release_id,
            operation="rollback",
            previous_release_id=(
                previous_release_id
            ),
        )
        pointer_changed = True

        # Activate the already validated bundle in this API process.
        result = activate_serving_bundle(
            candidate_bundle
        )

    except Exception as error:
        if pointer_changed:
            try:
                activate_release_pointer(
                    models_path=MODELS_PATH,
                    release_id=(
                        previous_release_id
                    ),
                    operation=(
                        "rollback_reverted"
                    ),
                    previous_release_id=(
                        payload.release_id
                    ),
                )

            except Exception:
                logger.exception(
                    "CRITICAL: rollback pointer "
                    "could not be restored | "
                    "expected_release_id=%s",
                    previous_release_id,
                )

        logger.exception(
            "Serving release rollback failed | "
            "target_release_id=%s | "
            "previous_release_id=%s | "
            "pointer_changed=%s",
            payload.release_id,
            previous_release_id,
            pointer_changed,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Serving release rollback failed. "
                f"Reason: {error}"
            ),
        ) from error

    logger.warning(
        "Serving release rollback completed | "
        "previous_release_id=%s | "
        "active_release_id=%s | "
        "model_version=%s",
        previous_release_id,
        candidate_bundle.release_id,
        candidate_bundle.model_version,
    )

    return {
        "status": "rolled_back",
        "previous_release_id": (
            previous_release_id
        ),
        **result,
    }



MAX_BATCH_ROWS = 5000


@app.post(
    "/predict",
    dependencies=[Depends(get_api_key)],
    response_model=PredictionResponse,
)
def predict(payload: PredictionRequest):
    request_started = time.perf_counter()
    timings: dict[str, float] = {}

    request_id = (
        payload.context.get("request_id")
        if (
            payload.context
            and payload.context.get(
                "request_id"
            )
        )
        else str(uuid4())
    )

    environment = os.getenv(
        "APP_ENV",
        "dev",
    )

    # Capture exactly one immutable serving snapshot for this request.
    request_bundle = active_serving_bundle

    if request_bundle is None:
        logger.error(
            "Predict called without an active serving bundle."
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "No complete serving bundle is active."
            ),
        )

    try:
        if len(payload.inputs) > MAX_BATCH_ROWS:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Batch too large. "
                    f"Max supported rows: "
                    f"{MAX_BATCH_ROWS}"
                ),
            )

        t = time.perf_counter()
        input_df = request_to_dataframe(
            payload.inputs
        )
        timings[
            "request_to_dataframe"
        ] = _ms_since(t)

        t = time.perf_counter()
        validated_input_df = (
            validate_prediction_input(
                input_df
            )
        )
        timings[
            "validate_prediction_input"
        ] = _ms_since(t)

        t = time.perf_counter()

        try:
            dq_summary = (
                log_data_quality_runtime(
                    validated_input_df,
                    reference_categories=(
                        dq_reference_categories
                    ),
                )
            )

        except Exception as dq_error:
            dq_summary = {
                "quality_status": "error",
                "error": str(dq_error),
            }
            logger.warning(
                "Data quality logging failed: %s",
                dq_error,
            )

        timings[
            "log_data_quality"
        ] = _ms_since(t)

        predictions: list[float] = []
        t_batch = time.perf_counter()

        for row in payload.inputs:
            row_df = request_to_dataframe(
                [row]
            )

            row_validated_df = (
                validate_prediction_input(
                    row_df
                )
            )

            store_id = (
                resolve_forecasting_store_id(
                    row_validated_df
                )
            )

            open_flags = resolve_open_flags(
                row_validated_df
            )

            # Every inference dependency comes from the same bundle.
            features_df = (
                merge_request_with_metadata(
                    validated_df=(
                        row_validated_df
                    ),
                    store_metadata=(
                        request_bundle
                        .store_metadata
                    ),
                    store_id=store_id,
                )
            )

            features_df = (
                merge_request_with_calendar(
                    features_df,
                    request_bundle
                    .known_calendar,
                )
            )

            processed_df = preprocess_data(
                features_df,
                mode="inference",
            )

            processed_df = (
                inject_forecasting_state_features(
                    processed_df=processed_df,
                    store_state=(
                        request_bundle
                        .store_state
                    ),
                    store_id=store_id,
                )
            )

            processed_df = (
                finalize_forecasting_feature_frame(
                    processed_df
                )
            )

            processed_df = (
                align_features_for_model(
                    processed_df=processed_df,
                    model=(
                        request_bundle.model
                    ),
                    model_type=(
                        request_bundle
                        .model_type
                    ),
                )
            )

            raw_predictions = (
                request_bundle.model.predict(
                    processed_df
                )
            )

            row_predictions = [
                float(
                    inverse_transform_target(
                        float(prediction),
                        request_bundle
                        .target_transformation,
                    )
                )
                for prediction
                in raw_predictions
            ]

            row_predictions = (
                apply_prediction_postprocessing(
                    row_predictions,
                    open_flags,
                )
            )

            predictions.extend(
                row_predictions
            )

        timings["predict_rows_single_logic"] = _ms_since(t_batch)

        t = time.perf_counter()
        rounded_predictions = [round(float(pred), 2) for pred in predictions]
        timings["postprocess_predictions"] = _ms_since(t)

        if len(rounded_predictions) != len(payload.inputs):
            raise RuntimeError(
                f"Prediction count mismatch: got {len(rounded_predictions)} predictions "
                f"for {len(payload.inputs)} input rows."
            )

        t = time.perf_counter()
        for features, pred in zip(
            payload.inputs,
            rounded_predictions,
        ):
            log_prediction(
                features,
                float(pred),
                release_id=request_bundle.release_id,
                model_alias=request_bundle.serving_alias,
                model_version=request_bundle.model_version,
                model_run_id=request_bundle.model_run_id,
                request_id=request_id,
                environment=environment,
            )
        timings["log_prediction"] = _ms_since(t)

        timings["total"] = _ms_since(request_started)

        logger.info(
            "Prediction completed",
            extra={
                "timing_ms": timings,
                "rows": len(rounded_predictions),
                "unique_stores": (
                    int(
                        validated_input_df[
                            "Store"
                        ].nunique()
                    )
                    if "Store"
                    in validated_input_df.columns
                    else None
                ),
                "path": "/predict",
                "release_id": (
                    request_bundle.release_id
                ),
                "model_type": (
                    request_bundle.model_type
                ),
                "model_version": (
                    request_bundle.model_version
                ),
                "model_run_id": (
                    request_bundle.model_run_id
                ),
                "serving_alias": (
                    request_bundle.serving_alias
                ),
                "request_id": request_id,
            },
        )

        return {
            "predictions": rounded_predictions,
            "status": "success",
            "metadata": {
                "rows": len(rounded_predictions),
                "unique_stores": (
                    int(
                        validated_input_df[
                            "Store"
                        ].nunique()
                    )
                    if "Store"
                    in validated_input_df.columns
                    else None
                ),
                "release_id": (
                    request_bundle.release_id
                ),
                "model_name": (
                    request_bundle.model_name
                ),
                "model_type": (
                    request_bundle.model_type
                ),
                "model_version": (
                    request_bundle.model_version
                ),
                "model_run_id": (
                    request_bundle.model_run_id
                ),
                "target_transformation": (
                    request_bundle.target_transformation
                ),
                "serving_alias": (
                    request_bundle.serving_alias
                ),
                "model_uri": (
                    request_bundle.model_uri
                ),
                "request_id": request_id,
                "timing_ms": timings,
                "data_quality": dq_summary,
            },
        }

    except HTTPException:
        timings["total"] = _ms_since(
            request_started
        )

        logger.error(
            "Prediction failed with HTTPException",
            extra={
                "timing_ms": timings,
                "path": "/predict",
                "request_id": request_id,
                "release_id": (
                    request_bundle.release_id
                ),
            },
        )

        raise

    except Exception as e:
        timings["total"] = _ms_since(request_started)

        logger.error(
            f"Prediction failed: {str(e)}",
            extra={
                "timing_ms": timings,
                "path": "/predict",
                "request_id": request_id,
            },
        )
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")