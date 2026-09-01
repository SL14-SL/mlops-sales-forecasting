from __future__ import annotations

import os
import time
from typing import Any
from uuid import uuid4

from src.api.middleware import (
    _ms_since,
)
from src.api.schema import (
    PredictionRequest,
)
from src.inference.prediction_service import (
    predict_with_bundle,
)
from src.inference.serving_bundle import (
    ServingBundle,
)
from src.monitoring.data_quality import (
    log_data_quality_runtime,
)
from src.monitoring.prediction_logger import (
    log_prediction,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _resolve_request_id(
    payload: PredictionRequest,
) -> str:
    if (
        payload.context
        and payload.context.get(
            "request_id"
        )
    ):
        return str(
            payload.context[
                "request_id"
            ]
        )

    return str(uuid4())


def _is_deployment_probe(
    payload: PredictionRequest,
) -> bool:
    return bool(
        payload.context
        and payload.context.get(
            "purpose"
        )
        == (
            "post_deployment_verification"
        )
    )


def _log_data_quality(
    *,
    validated_input,
    reference_categories: (
        dict[str, set[str]]
    ),
) -> dict[str, Any]:
    try:
        return log_data_quality_runtime(
            validated_input,
            reference_categories=(
                reference_categories
            ),
        )

    except Exception as error:
        logger.warning(
            "Data quality logging "
            "failed: %s",
            error,
        )

        return {
            "quality_status": "error",
            "error": str(error),
        }


def _log_production_predictions(
    *,
    inputs: list[dict[str, Any]],
    predictions: list[float],
    bundle: ServingBundle,
    request_id: str,
    environment: str,
) -> None:
    for features, prediction in zip(
        inputs,
        predictions,
    ):
        log_prediction(
            features,
            float(prediction),
            release_id=(
                bundle.release_id
            ),
            model_alias=(
                bundle.serving_alias
            ),
            model_version=(
                bundle.model_version
            ),
            model_run_id=(
                bundle.model_run_id
            ),
            request_id=request_id,
            environment=environment,
        )


def _build_prediction_response(
    *,
    predictions: list[float],
    bundle: ServingBundle,
    request_id: str,
    unique_stores: int | None,
    timings: dict[str, float],
    data_quality: dict[str, Any],
    is_deployment_probe: bool,
) -> dict[str, Any]:
    """
    Build an API response from prediction output and serving metadata.
    """
    return {
        "predictions": predictions,
        "status": "success",
        "metadata": {
            "rows": len(predictions),
            "unique_stores": (
                unique_stores
            ),
            "release_id": (
                bundle.release_id
            ),
            "model_name": (
                bundle.model_name
            ),
            "model_type": (
                bundle.model_type
            ),
            "model_version": (
                bundle.model_version
            ),
            "model_run_id": (
                bundle.model_run_id
            ),
            "target_transformation": (
                bundle
                .target_transformation
            ),
            "serving_alias": (
                bundle.serving_alias
            ),
            "model_uri": (
                bundle.model_uri
            ),
            "request_id": request_id,
            "timing_ms": timings,
            "data_quality": (
                data_quality
            ),
            "deployment_probe": (
                is_deployment_probe
            ),
        },
    }


def handle_prediction(
    *,
    payload: PredictionRequest,
    bundle: ServingBundle,
    reference_categories: (
        dict[str, set[str]]
    ),
) -> dict[str, Any]:
    """
    Execute a prediction request and perform all associated monitoring work.

    The handler resolves request metadata, evaluates runtime data quality, invokes
    the active serving bundle, logs production predictions and constructs the API
    response.

    Args:
        request: Validated forecasting request.
        serving_bundle: Immutable model and inference-artifact bundle.
        request_id: Optional externally supplied request identifier.

    Returns:
        The complete prediction response including model and release lineage.

    Raises:
        ValueError: If request data cannot be converted into valid model input.
        RuntimeError: If prediction execution or response construction fails.
    """
    request_started = (
        time.perf_counter()
    )
    timings: dict[str, float] = {}

    request_id = _resolve_request_id(
        payload
    )

    environment = os.getenv(
        "APP_ENV",
        "dev",
    )

    is_deployment_probe = (
        _is_deployment_probe(
            payload
        )
    )

    inference_result = (
        predict_with_bundle(
            inputs=payload.inputs,
            bundle=bundle,
        )
    )

    predictions = list(
        inference_result.predictions
    )

    timings.update(
        inference_result.timings_ms
    )

    started_at = time.perf_counter()

    data_quality = _log_data_quality(
        validated_input=(
            inference_result
            .validated_input
        ),
        reference_categories=(
            reference_categories
        ),
    )

    timings[
        "log_data_quality"
    ] = _ms_since(
        started_at
    )

    started_at = time.perf_counter()

    if not is_deployment_probe:
        _log_production_predictions(
            inputs=payload.inputs,
            predictions=predictions,
            bundle=bundle,
            request_id=request_id,
            environment=environment,
        )

    timings[
        "log_prediction"
    ] = _ms_since(
        started_at
    )

    timings["total"] = _ms_since(
        request_started
    )

    logger.info(
        "Prediction completed",
        extra={
            "timing_ms": timings,
            "rows": len(predictions),
            "deployment_probe": (
                is_deployment_probe
            ),
            "unique_stores": (
                inference_result
                .unique_stores
            ),
            "path": "/predict",
            "release_id": (
                bundle.release_id
            ),
            "model_type": (
                bundle.model_type
            ),
            "model_version": (
                bundle.model_version
            ),
            "model_run_id": (
                bundle.model_run_id
            ),
            "serving_alias": (
                bundle.serving_alias
            ),
            "request_id": request_id,
        },
    )

    return _build_prediction_response(
        predictions=predictions,
        bundle=bundle,
        request_id=request_id,
        unique_stores=(
            inference_result
            .unique_stores
        ),
        timings=timings,
        data_quality=data_quality,
        is_deployment_probe=(
            is_deployment_probe
        ),
    )