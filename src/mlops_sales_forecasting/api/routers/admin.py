from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ...inference.model_manager import ModelManager
from ..dependencies import (
    get_model_manager,
    require_api_key,
)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
)


@router.post(
    "/reload",
    response_model=None,
)
def reload_serving_bundle(
    _: Annotated[
        None,
        Depends(require_api_key),
    ],
    model_manager: Annotated[
        ModelManager | None,
        Depends(get_model_manager),
    ],
) -> dict[str, Any] | JSONResponse:
    """Reload the active serving bundle."""
    if model_manager is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "model_manager_unavailable",
            },
        )

    result = model_manager.reload()

    if not result.success:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "previous_release_id": (
                    result.previous_release_id
                ),
                "active_release_id": (
                    result.active_release_id
                ),
                "error": result.error,
            },
        )

    return {
        "status": "success",
        "previous_release_id": (
            result.previous_release_id
        ),
        "active_release_id": (
            result.active_release_id
        ),
    }