from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from mlops_sales_forecasting.api.app import create_app
from mlops_sales_forecasting.inference.model_manager import (
    ReloadResult,
)

API_KEY = "test-secret-api-key"


def build_client(
    model_manager=None,
    *,
    api_key: str | None = API_KEY,
) -> TestClient:
    app = create_app(
        model_manager=model_manager,
        load_model_on_startup=False,
        api_key=api_key,
    )
    return TestClient(app)


def test_reload_requires_api_key() -> None:
    with build_client(MagicMock()) as client:
        response = client.post("/admin/reload")

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "API key is required."
    )


def test_reload_rejects_invalid_api_key() -> None:
    with build_client(MagicMock()) as client:
        response = client.post(
            "/admin/reload",
            headers={
                "X-API-Key": "wrong-key",
            },
        )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Invalid API key."
    )


def test_reload_requires_configured_authentication() -> None:
    with build_client(
        MagicMock(),
        api_key=None,
    ) as client:
        response = client.post(
            "/admin/reload",
            headers={
                "X-API-Key": API_KEY,
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "API authentication is not configured."
    )


def test_reload_requires_model_manager() -> None:
    with build_client(None) as client:
        response = client.post(
            "/admin/reload",
            headers={
                "X-API-Key": API_KEY,
            },
        )

    assert response.status_code == 503
    assert response.json()["error"] == (
        "model_manager_unavailable"
    )


def test_reload_returns_success() -> None:
    model_manager = MagicMock()
    model_manager.reload.return_value = ReloadResult(
        success=True,
        previous_release_id="release-1",
        active_release_id="release-2",
    )

    with build_client(model_manager) as client:
        response = client.post(
            "/admin/reload",
            headers={
                "X-API-Key": API_KEY,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "previous_release_id": "release-1",
        "active_release_id": "release-2",
    }


def test_failed_reload_preserves_active_release() -> None:
    model_manager = MagicMock()
    model_manager.reload.return_value = ReloadResult(
        success=False,
        previous_release_id="release-1",
        active_release_id="release-1",
        error="invalid serving release",
    )

    with build_client(model_manager) as client:
        response = client.post(
            "/admin/reload",
            headers={
                "X-API-Key": API_KEY,
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "previous_release_id": "release-1",
        "active_release_id": "release-1",
        "error": "invalid serving release",
    }