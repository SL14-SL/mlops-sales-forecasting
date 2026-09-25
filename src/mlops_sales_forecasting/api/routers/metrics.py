from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

router = APIRouter(tags=["monitoring"])


@router.get(
    "/metrics",
    include_in_schema=False,
)
def metrics() -> Response:
    """Expose metrics in Prometheus text format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )