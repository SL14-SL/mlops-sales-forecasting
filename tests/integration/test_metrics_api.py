from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from mlops_sales_forecasting.api.app import create_app
from mlops_sales_forecasting.inference.model_manager import (
    ReloadResult,
)


def test_metrics_endpoint_exposes_prometheus_format() -> None:
    app = create_app(
        model_manager=None,
        load_model_on_startup=False,
    )

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "mlops_api_requests_total" in response.text


def test_admin_request_is_recorded() -> None:
    model_manager = MagicMock()
    model_manager.reload.return_value = ReloadResult(
        success=True,
        previous_release_id="release-1",
        active_release_id="release-2",
    )
    app = create_app(
        model_manager=model_manager,
        load_model_on_startup=False,
        api_key="test-key",
    )

    with TestClient(app) as client:
        client.post(
            "/admin/reload",
            headers={
                "X-API-Key": "test-key",
            },
        )
        metrics_response = client.get("/metrics")

    assert (
        'mlops_api_requests_total{'
        'method="POST",path="/admin/reload",status_code="200"}'
        in metrics_response.text
    )