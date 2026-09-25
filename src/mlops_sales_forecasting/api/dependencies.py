import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from ..inference.model_manager import ModelManager


def get_model_manager(
    request: Request,
) -> ModelManager | None:
    """Return the application's model manager."""
    return getattr(
        request.app.state,
        "model_manager",
        None,
    )


def get_expected_api_key(
    request: Request,
) -> str | None:
    """Return the configured API key."""
    return getattr(
        request.app.state,
        "api_key",
        None,
    )


def require_api_key(
    request: Request,
    provided_api_key: Annotated[
        str | None,
        Header(alias="X-API-Key"),
    ] = None,
) -> None:
    """Require a valid API key for a protected endpoint."""
    expected_api_key = get_expected_api_key(request)

    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if provided_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required.",
        )

    if not secrets.compare_digest(
        provided_api_key,
        expected_api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )