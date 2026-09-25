from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ...inference.model_manager import ModelManager
from ..dependencies import get_model_manager

router = APIRouter(tags=["health"])


@router.get("/livez")
def liveness() -> dict[str, str]:
    """Report whether the API process is running."""
    return {
        "status": "live",
    }


@router.get(
    "/readyz",
    response_model=None,
)
def readiness(
    model_manager: Annotated[
        ModelManager | None,
        Depends(get_model_manager),
    ],
) -> dict[str, Any] | JSONResponse:
    """Report whether a serving bundle is available."""
    if model_manager is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "model_manager_unavailable",
                "active_release_id": None,
            },
        )

    if not model_manager.ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": (
                    model_manager.last_reload_error
                    or "serving_bundle_unavailable"
                ),
                "active_release_id": (
                    model_manager.active_release_id
                ),
            },
        )

    return {
        "status": "ready",
        "active_release_id": (
            model_manager.active_release_id
        ),
    }