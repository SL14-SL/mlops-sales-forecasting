import time
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)

from ...inference.model_manager import (
    ModelManager,
    ModelNotReadyError,
)
from ...inference.prediction_service import PredictionService
from ...inference.serving_bundle import ServingBundle
from ...monitoring.model_observability import observe_model_outputs
from ...monitoring.prediction import observe_prediction
from ...monitoring.prediction_event_logger import (
    PredictionEvent,
    log_prediction_event,
)
from ...utils.logger import get_logger
from ..dependencies import (
    get_application_config,
    get_model_manager,
    require_api_key,
)
from ..schema import (
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["prediction"])


def _request_id(request: Request) -> str:
    """Return the request ID created by the API middleware."""
    return str(
        getattr(
            request.state,
            "request_id",
            "unknown",
        )
    )


def _record_prediction_event(
    *,
    request: Request,
    prediction_request: PredictionRequest,
    bundle: ServingBundle,
    started_at: float,
    error: Exception | None = None,
) -> None:
    """Record privacy-safe logs and operational metrics."""
    duration_seconds = (
        time.perf_counter() - started_at
    )
    prediction_status = (
        "error"
        if error is not None
        else "success"
    )
    task_type = bundle.manifest.task_type.value
    observation_count = len(
        prediction_request.inputs
    )

    try:
        event = PredictionEvent(
            request_id=_request_id(request),
            release_id=bundle.release_id,
            task_type=task_type,
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            batch_size=observation_count,
            duration_ms=(
                duration_seconds * 1000
            ),
            status=prediction_status,
            error_type=(
                type(error).__name__
                if error is not None
                else None
            ),
        )
        log_prediction_event(event)
    except Exception:
        logger.exception(
            "Could not record prediction event | request_id=%s",
            _request_id(request),
        )

    try:
        observe_prediction(
            task_type=task_type,
            status=prediction_status,
            observation_count=observation_count,
            latency_seconds=duration_seconds,
        )
    except Exception:
        logger.exception(
            "Could not record prediction metrics | request_id=%s",
            _request_id(request),
        )

def _record_model_outputs(
    *,
    response: PredictionResponse,
    request: Request,
) -> None:
    """Record model outputs without affecting the response."""
    try:
        observe_model_outputs(response)
    except Exception:
        logger.exception(
            "Could not record model output metrics | request_id=%s",
            _request_id(request),
        )

@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    http_request: Request,
    request: PredictionRequest,
    _: Annotated[
        None,
        Depends(require_api_key),
    ],
    model_manager: Annotated[
        ModelManager | None,
        Depends(get_model_manager),
    ],
    application_config: Annotated[
        dict[str, Any],
        Depends(get_application_config),
    ],
) -> PredictionResponse:
    """Run a prediction using the active serving bundle."""
    if model_manager is None or not model_manager.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready.",
        )

    try:
        bundle = model_manager.get_bundle()
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready.",
        ) from exc

    service = PredictionService(
        model_manager,
        application_config,
    )
    started_at = time.perf_counter()

    try:
        response = service.predict(request)
    except ModelNotReadyError as exc:
        _record_prediction_event(
            request=http_request,
            prediction_request=request,
            bundle=bundle,
            started_at=started_at,
            error=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready.",
        ) from exc
    except ValueError as exc:
        _record_prediction_event(
            request=http_request,
            prediction_request=request,
            bundle=bundle,
            started_at=started_at,
            error=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _record_prediction_event(
            request=http_request,
            prediction_request=request,
            bundle=bundle,
            started_at=started_at,
            error=exc,
        )
        raise

    _record_model_outputs(
        response=response,
        request=http_request,
    )

    _record_prediction_event(
        request=http_request,
        prediction_request=request,
        bundle=bundle,
        started_at=started_at,
    )
    return response