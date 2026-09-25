from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from mlops_sales_forecasting.api.app import create_app


def manager(
    *,
    ready: bool,
    release_id: str | None = None,
    error: str | None = None,
) -> MagicMock:
    result = MagicMock()
    result.ready = ready
    result.active_release_id = release_id
    result.last_reload_error = error
    return result


def test_liveness_without_model_manager() -> None:
    app = create_app(
        model_manager=None,
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {
        "status": "live",
    }


def test_readiness_without_model_manager() -> None:
    app = create_app(
        model_manager=None,
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["reason"] == (
        "model_manager_unavailable"
    )


def test_readiness_with_active_bundle() -> None:
    model_manager = manager(
        ready=True,
        release_id="release-5",
    )
    app = create_app(
        model_manager=model_manager,
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "active_release_id": "release-5",
    }


def test_readiness_exposes_reload_error() -> None:
    model_manager = manager(
        ready=False,
        error="MLflow unavailable",
    )
    app = create_app(
        model_manager=model_manager,
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["reason"] == "MLflow unavailable"


def test_startup_loads_initial_bundle() -> None:
    model_manager = manager(
        ready=True,
        release_id="release-1",
    )
    app = create_app(
        model_manager=model_manager,
        load_model_on_startup=True,
    )

    with TestClient(app) as client:
        response = client.get("/livez")

    assert response.status_code == 200
    model_manager.load_initial.assert_called_once_with()


def test_failed_startup_keeps_api_live() -> None:
    model_manager = manager(
        ready=False,
        error="active release unavailable",
    )
    model_manager.load_initial.side_effect = RuntimeError(
        "active release unavailable"
    )
    app = create_app(
        model_manager=model_manager,
        load_model_on_startup=True,
    )

    with TestClient(app) as client:
        live_response = client.get("/livez")
        ready_response = client.get("/readyz")

    assert live_response.status_code == 200
    assert ready_response.status_code == 503

def test_application_uses_custom_title() -> None:
    app = create_app(
        model_manager=None,
        load_model_on_startup=False,
        title="Customer Churn API",
    )

    assert app.title == "Customer Churn API"