import os
import traceback
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import mlflow
from fastapi import FastAPI, HTTPException, Security, Depends, Response, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse, JSONResponse

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.status import HTTP_403_FORBIDDEN

from src.api.schema import PredictionRequest, PredictionResponse
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
    load_store_metadata,
    load_store_state,
)
from src.data.features.build_features import preprocess_data

from src.inference.forecasting_policy import (
    merge_request_with_metadata,
    inject_forecasting_state_features,
    finalize_forecasting_feature_frame,
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
MODELS_PATH = Path(get_path("models"))
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles setup and teardown of the forecasting API.

    Startup responsibilities:
    - optionally skip heavy loading in smoke-test mode
    - load store metadata
    - load forecasting state snapshot
    - load current champion model from MLflow registry

    The API is resilient by design:
    - missing metadata/state/model does not crash startup
    - readiness endpoints report degraded state
    """
    global model, store_metadata, store_state, model_type, target_transformation
    global model_uri, serving_alias, serving_model_version, serving_model_run_id
    global dq_reference_categories

    if os.getenv("SMOKE_TEST") == "1":
        logger.info(
            "Smoke test mode enabled. Skipping model, metadata and state startup loading."
        )
        yield
        return

    try:
        # -------------------------------------------------
        # 1. Load store metadata
        # -------------------------------------------------
        try:
            store_metadata = load_store_metadata(
                validated_path=VALIDATED_PATH,
                gcs_bucket=GCS_BUCKET,
            )

            if store_metadata is None:
                logger.warning(
                    "Store metadata could not be loaded. "
                    "API will start in degraded mode."
                )
            else:
                logger.info("Store metadata loaded successfully.")

        except Exception as metadata_error:
            logger.warning(
                "Store metadata loading failed: %s. API will start in degraded mode.",
                metadata_error,
            )
            store_metadata = None

        # -------------------------------------------------
        # 2. Initialize data quality reference cache
        # -------------------------------------------------
        try:
            ref_df = initialize_data_quality_reference_cache()
            dq_reference_categories = build_reference_category_cache(
                ref_df,
                categorical_reference_features=get_data_quality_settings().get(
                    "categorical_reference_features", []
                ),
            )
            logger.info("Data quality reference cache initialized.")

        except Exception as dq_error:
            logger.warning(
                "Data quality reference cache initialization failed: %s. "
                "Continuing with empty reference categories.",
                dq_error,
            )
            dq_reference_categories = {}

        # -------------------------------------------------
        # 3. Load forecasting state snapshot
        # -------------------------------------------------
        try:
            store_state = load_store_state(
                models_path=MODELS_PATH,
                gcs_bucket=GCS_BUCKET,
            )

            if store_state is None:
                store_state = {}

            logger.info("Forecasting state loaded successfully.")

        except Exception as state_error:
            logger.warning(
                "Forecasting state loading failed: %s. Continuing with empty state.",
                state_error,
            )
            store_state = {}

        # -------------------------------------------------
        # 4. Load current champion model from MLflow
        # -------------------------------------------------
        try:
            reload_result = reload_serving_model()

            logger.info(
                "Model loaded from MLflow registry. "
                "alias=%s | version=%s | run_id=%s | model_type=%s | target_transformation=%s",
                reload_result.get("serving_alias"),
                reload_result.get("model_version"),
                reload_result.get("model_run_id"),
                reload_result.get("model_type"),
                reload_result.get("target_transformation"),
            )

        except Exception as model_error:
            logger.error(
                "Registry model load failed: %s. API will start in degraded mode.",
                model_error,
            )
            model = None
            serving_alias = "unavailable"
            model_uri = None
            serving_model_version = None
            serving_model_run_id = None

        logger.info("Startup sequence finished. API listening.")
        yield

    except Exception as critical_error:
        logger.error("Critical startup error: %s", critical_error)
        logger.error(traceback.format_exc())
        yield

    finally:
        logger.info("Shutdown: cleaning up API assets.")

app = FastAPI(title="Blueprint Demand Forecasting API", lifespan=lifespan)

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
    return {
        "status": "alive",
        "service": CFG.get("project_name", "sales-forecasting-api"),
        "environment": CFG.get("environment", "unknown"),
    }


@app.get("/readyz")
def readyz():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if store_metadata is None:
        raise HTTPException(status_code=503, detail="Store metadata is not loaded.")

    if store_state is None:
        raise HTTPException(status_code=503, detail="Forecasting state is not loaded.")

    return {
        "status": "ready",
        "model_name": MODEL_NAME,
        "model_type": model_type,
        "target_transformation": target_transformation,
        "serving_alias": serving_alias,
        "model_version": serving_model_version,
        "model_run_id": serving_model_run_id,
        "model_uri": model_uri,
        "store_metadata_loaded": store_metadata is not None,
        "state_loaded": store_state is not None,
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
def reload_serving_state(api_key: str = Depends(get_api_key)):
    """
    Reload forecasting serving state:
    - champion model
    - store metadata
    - forecasting state snapshot
    """
    global store_metadata, store_state

    try:
        model_result = reload_serving_model()

        store_metadata = load_store_metadata(
            validated_path=VALIDATED_PATH,
            gcs_bucket=GCS_BUCKET,
        )

        store_state = load_store_state(
            models_path=MODELS_PATH,
            gcs_bucket=GCS_BUCKET,
        )

    except Exception as error:
        logger.error("Serving state reload failed: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Serving state reload failed: {str(error)}",
        )

    return {
        "status": "reloaded",
        "store_metadata_loaded": store_metadata is not None,
        "state_loaded": store_state is not None,
        **model_result,
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
        "model_type": model_type,
        "target_transformation": target_transformation,
        "model_name": MODEL_NAME,
        "tracking_uri": mlflow.get_tracking_uri(),
        "serving_alias": serving_alias,
        "model_uri": model_uri,
        "model_version": serving_model_version,
        "model_run_id": serving_model_run_id,
    }

MAX_BATCH_ROWS = 5000


@app.post(
    "/predict",
    dependencies=[Depends(get_api_key)],
    response_model=PredictionResponse,
)
def predict(payload: PredictionRequest):
    if model is None or store_metadata is None:
        logger.error("Predict called but model/metadata is missing!")
        raise HTTPException(
            status_code=503,
            detail="Model or metadata not ready. Ensure '@champion' alias is set in MLflow or local fallback is available.",
        )

    request_started = time.perf_counter()
    timings: dict[str, float] = {}

    request_id = (
        payload.context.get("request_id")
        if payload.context and payload.context.get("request_id")
        else str(uuid4())
    )
    environment = os.getenv("APP_ENV", "dev")

    try:
        if len(payload.inputs) > MAX_BATCH_ROWS:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large. Max supported rows: {MAX_BATCH_ROWS}",
            )

        t = time.perf_counter()
        input_df = request_to_dataframe(payload.inputs)
        timings["request_to_dataframe"] = _ms_since(t)

        t = time.perf_counter()
        validated_input_df = validate_prediction_input(input_df)
        timings["validate_prediction_input"] = _ms_since(t)

        t = time.perf_counter()
        try:
            dq_summary = log_data_quality_runtime(
                validated_input_df,
                reference_categories=dq_reference_categories,
            )
        except Exception as dq_error:
            dq_summary = {"quality_status": "error", "error": str(dq_error)}
            logger.warning(f"Data quality logging failed: {dq_error}")
        timings["log_data_quality"] = _ms_since(t)

        predictions: list[float] = []

        t_batch = time.perf_counter()

        for row in payload.inputs:
            # Alte Single-Request-Semantik pro Zeile
            row_df = request_to_dataframe([row])
            row_validated_df = validate_prediction_input(row_df)

            store_id = resolve_forecasting_store_id(row_validated_df)
            open_flags = resolve_open_flags(row_validated_df)

            # Alte Single-Store-Logik
            features_df = merge_request_with_metadata(
                validated_df=row_validated_df,
                store_metadata=store_metadata,
                store_id=store_id,
            )

            processed_df = preprocess_data(features_df, mode="inference")

            processed_df = inject_forecasting_state_features(
                processed_df=processed_df,
                store_state=store_state or {},
                store_id=store_id,
            )

            processed_df = finalize_forecasting_feature_frame(processed_df)

            processed_df = align_features_for_model(
                processed_df=processed_df,
                model=model,
                model_type=model_type,
            )

            raw_pred = model.predict(processed_df)

            row_predictions = [
                float(inverse_transform_target(float(pred), target_transformation))
                for pred in raw_pred
            ]

            row_predictions = apply_prediction_postprocessing(
                row_predictions,
                open_flags,
            )

            predictions.extend(row_predictions)

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
        for features, pred in zip(payload.inputs, rounded_predictions):
            log_prediction(
                features,
                float(pred),
                model_alias=serving_alias,
                model_version=serving_model_version,
                model_run_id=serving_model_run_id,
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
                "unique_stores": int(validated_input_df["Store"].nunique())
                if "Store" in validated_input_df.columns
                else None,
                "path": "/predict",
                "model_type": model_type,
                "serving_alias": serving_alias,
                "request_id": request_id,
            },
        )

        return {
            "predictions": rounded_predictions,
            "status": "success",
            "metadata": {
                "rows": len(rounded_predictions),
                "unique_stores": int(validated_input_df["Store"].nunique())
                if "Store" in validated_input_df.columns
                else None,
                "model_name": MODEL_NAME,
                "model_type": model_type,
                "target_transformation": target_transformation,
                "serving_alias": serving_alias,
                "model_uri": model_uri,
                "request_id": request_id,
                "timing_ms": timings,
                "data_quality": dq_summary,
            },
        }

    except HTTPException:
        timings["total"] = _ms_since(request_started)
        logger.error(
            "Prediction failed with HTTPException",
            extra={
                "timing_ms": timings,
                "path": "/predict",
                "request_id": request_id,
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