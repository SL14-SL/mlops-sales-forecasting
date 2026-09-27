from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ...inference.model_manager import ModelManager
from ...monitoring.summary import (
    build_monitoring_summary,
)
from ..dependencies import (
    get_application_config,
    get_model_manager,
)

router = APIRouter(tags=["monitoring"])


@router.get(
    "/monitoring/summary",
    include_in_schema=False,
)
def monitoring_summary(
    application_config: Annotated[
        dict[str, Any],
        Depends(get_application_config),
    ],
    model_manager: Annotated[
        ModelManager | None,
        Depends(get_model_manager),
    ],
) -> dict[str, Any]:
    """Return persisted operational monitoring state."""
    return build_monitoring_summary(
        config=application_config,
        model_manager=model_manager,
    )
