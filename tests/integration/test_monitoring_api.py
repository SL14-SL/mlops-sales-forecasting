from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from mlops_sales_forecasting.api.app import create_app


def test_monitoring_summary_endpoint(
    tmp_path: Path,
) -> None:
    manager = MagicMock()
    manager.ready = True
    manager.active_release_id = "release-7"
    manager.last_reload_error = None

    app = create_app(
        model_manager=manager,
        load_model_on_startup=False,
        config={
            "paths": {
                "monitoring": str(tmp_path / "monitoring"),
            },
        },
    )

    with TestClient(app) as client:
        response = client.get("/monitoring/summary")

    assert response.status_code == 200

    payload = response.json()

    assert payload["serving"] == {
        "ready": True,
        "active_release_id": "release-7",
        "last_reload_error": None,
    }
    assert payload["performance"] == {
        "available": False,
    }
    assert payload["feature_drift"] == {
        "available": False,
    }
    assert payload["retraining"] == {
        "available": False,
    }
