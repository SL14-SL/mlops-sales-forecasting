from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from ..inference.model_manager import ModelManager
from ..utils.logger import get_logger
from .middleware import request_context_middleware
from .routers.admin import router as admin_router
from .routers.health import router as health_router
from .routers.metrics import router as metrics_router
from .routers.prediction import router as prediction_router

logger = get_logger(__name__)


def create_app(
    *,
    model_manager: ModelManager | None = None,
    load_model_on_startup: bool = True,
    title: str = "MLOps Model API",
    api_key: str | None = None,
    config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        if (
            model_manager is not None
            and load_model_on_startup
        ):
            try:
                model_manager.load_initial()
            except Exception as exc:
                logger.error(
                    "Initial serving bundle load failed: %s",
                    exc,
                )

        yield

    app = FastAPI(
        title=title,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.model_manager = model_manager
    app.state.api_key = api_key
    app.state.config = dict(
        config or {}
    )
    app.middleware("http")(request_context_middleware)
    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(metrics_router)
    app.include_router(prediction_router)

    return app