import pandas as pd
import pytest
from unittest.mock import patch
from unittest.mock import MagicMock
import sys

from src.data.features.calendar import build_known_calendar, prepare_known_calendar_lookup


@pytest.fixture(autouse=True)
def mock_api_dependencies(
    monkeypatch,
    mock_xgb_model,
    sample_store_metadata,
    sample_store_state,
):
    monkeypatch.setenv(
        "API_KEY",
        "test-secret-key",
    )
    monkeypatch.setenv(
        "APP_ENV",
        "dev",
    )

    mocked_processed_df = pd.DataFrame(
        [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Customers": 500,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
                "StoreType": "a",
                "Assortment": "c",
                "CompetitionDistance": 1200.0,
                "Promo2": 0,
                "WeekOfYear": 9,
                "day": 27,
                "month": 2,
                "year": 2026,
                "is_month_start": 0,
                "is_month_end": 0,
                "sales_lag_1": 1000.0,
                "sales_lag_7": 950.0,
                "sales_rolling_mean_7": 980.0,
            }
        ]
    )

    mocked_bundle = MagicMock()
    mocked_bundle.release_id = (
        "release-test-v1"
    )
    mocked_bundle.model = mock_xgb_model
    mocked_bundle.model_name = (
        "sales-forecasting-model-dev"
    )
    mocked_bundle.model_type = "xgboost"
    mocked_bundle.target_transformation = "log1p"
    mocked_bundle.serving_alias = "champion"
    mocked_bundle.model_uri = (
        "models:/sales-forecasting-model-dev@champion"
    )
    mocked_bundle.model_version = "1"
    mocked_bundle.model_run_id = "test-run-1"
    mocked_bundle.store_metadata = sample_store_metadata
    mocked_bundle.store_state = sample_store_state
    calendar_source = pd.DataFrame(
        {
            "Store": [
                1,
                1,
                1,
            ],
            "Date": pd.to_datetime(
                [
                    "2015-07-31",
                    "2026-02-27",
                    "2026-03-06",
                ]
            ),
            "StateHoliday": [
                "0",
                "0",
                "0",
            ],
            "SchoolHoliday": [
                1,
                0,
                0,
            ],
        }
    )

    mocked_bundle.known_calendar = (
        prepare_known_calendar_lookup(
            build_known_calendar(
                calendar_source
            )
        )
    )

    mock_log_prediction = MagicMock()

    with (
        patch(
            "src.api.app.active_serving_bundle",
            mocked_bundle,
        ),
        patch(
            "src.api.app.model",
            mock_xgb_model,
        ),
        patch(
            "src.api.app.store_metadata",
            sample_store_metadata,
        ),
        patch(
            "src.api.app.store_state",
            sample_store_state,
        ),
        patch(
            "src.api.app.model_type",
            "xgboost",
        ),
        patch(
            "src.api.app.target_transformation",
            "log1p",
        ),
        patch(
            "src.api.app.preprocess_data",
            return_value=mocked_processed_df,
        ),
        patch(
            "src.api.app.align_features_for_model",
            return_value=mocked_processed_df,
        ),
        patch(
            "src.api.app.log_prediction",
            mock_log_prediction,
        ),
    ):
        yield mock_log_prediction
        


def test_api_health_endpoint(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "online"
    assert body["model_loaded"] is True
    assert body["store_metadata_loaded"] is True
    assert body["model_type"] == "xgboost"
    assert body["target_transformation"] == "log1p"


def test_predict_endpoint_validation_error(api_client, api_headers):
    bad_payload = {"inputs": []}
    response = api_client.post("/predict", json=bad_payload, headers=api_headers)

    assert response.status_code == 422


def test_predict_endpoint_logic_error(api_client, api_headers):
    logic_error_payload = {
        "inputs": [
            {
                "Store": -1,
                "Date": "2026-02-27",
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
                "StoreType": "a",
                "Assortment": "c",
                "CompetitionDistance": 1200.0,
            }
        ]
    }

    response = api_client.post("/predict", json=logic_error_payload, headers=api_headers)

    assert response.status_code in (400, 422)


def test_predict_endpoint_success(api_client, api_headers, sample_prediction_payload):
    response = api_client.post("/predict", json=sample_prediction_payload, headers=api_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1
    assert "metadata" in body
    assert body["metadata"]["rows"] == 1
    assert body["metadata"]["release_id"] == (
    "release-test-v1"
    )
    assert body["metadata"]["model_version"] == "1"
    assert body["metadata"]["model_run_id"] == (
        "test-run-1"
    )


def test_predict_endpoint_requires_api_key(api_client, sample_prediction_payload):
    response = api_client.post("/predict", json=sample_prediction_payload)

    assert response.status_code == 403


def test_metrics_endpoint_exposes_custom_metrics(api_client):
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert "api_request_count_total" in response.text
    assert "api_request_latency_seconds" in response.text


def test_summary_endpoint_returns_human_readable_monitoring(api_client):
    api_client.get("/health")

    response = api_client.get("/monitoring/summary")
    assert response.status_code == 200

    payload = response.json()
    assert "requests_total" in payload
    assert "success_rate" in payload
    assert "latency_ms" in payload
    assert "status_codes" in payload


def test_metrics_endpoint_is_not_self_counted(api_client):
    api_client.get("/metrics")
    api_client.get("/metrics")

    summary = api_client.get("/monitoring/summary").json()

    # /metrics soll ignoriert werden, also keine Requests dafür im Summary-Path-Zähler
    assert "/metrics" not in summary["paths"]


def test_health_is_counted_as_success(api_client):
    response = api_client.get("/health")
    assert response.status_code in (200, 503)

    summary = api_client.get("/monitoring/summary").json()
    assert summary["requests_total"] >= 1
    assert "/health" in summary["paths"]


def test_unknown_route_is_counted_as_error(api_client):
    response = api_client.get("/does-not-exist")
    assert response.status_code == 404

    summary = api_client.get("/monitoring/summary").json()
    assert summary["error_total"] >= 1
    assert "404" in summary["status_codes"]


def test_predict_success_is_counted(api_client, api_headers):
    payload = {
        "inputs": [
            {
                "Store": 1,
                "Date": "2015-07-31",
                "Customers": 555,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 1,
            }
        ]
    }

    response = api_client.post("/predict", json=payload, headers=api_headers)
    assert response.status_code == 200

    summary = api_client.get("/monitoring/summary").json()
    assert "/predict" in summary["paths"]
    assert summary["success_total"] >= 1

def test_failed_bundle_reload_keeps_previous_serving_state(
    monkeypatch,
):
    previous_bundle = MagicMock()
    previous_bundle.model_version = "7"
    app_module = sys.modules["src.api.app"]

    monkeypatch.setattr(
        app_module,
        "active_serving_bundle",
        previous_bundle,
    )

    monkeypatch.setattr(
        app_module,
        "load_serving_bundle",
        MagicMock(
            side_effect=RuntimeError(
                "known calendar unavailable"
            )
        ),
    )

    with pytest.raises(RuntimeError):
        app_module.reload_complete_serving_bundle()

    assert (
        app_module.active_serving_bundle
        is previous_bundle
    )
    assert (
        app_module.active_serving_bundle.model_version
        == "7"
    )

def test_complete_bundle_is_activated_only_after_success(
    monkeypatch,
):

    app_module = sys.modules["src.api.app"]

    candidate_bundle = MagicMock()
    candidate_bundle.model = MagicMock()
    candidate_bundle.model_name = "sales-forecasting-model-dev"
    candidate_bundle.model_type = "xgboost"
    candidate_bundle.target_transformation = "log1p"
    candidate_bundle.serving_alias = "champion"
    candidate_bundle.model_uri = "models:/model@champion"
    candidate_bundle.model_version = "8"
    candidate_bundle.model_run_id = "run-8"
    candidate_bundle.store_metadata = MagicMock()
    candidate_bundle.store_state = {"1": [10.0]}
    candidate_bundle.known_calendar = MagicMock()

    monkeypatch.setattr(
        app_module,
        "load_serving_bundle",
        MagicMock(return_value=candidate_bundle),
    )

    result = app_module.reload_complete_serving_bundle()

    assert app_module.active_serving_bundle is candidate_bundle
    assert app_module.model is candidate_bundle.model
    assert app_module.serving_model_version == "8"
    assert result["model_version"] == "8"

def test_readyz_returns_503_without_active_bundle(
    api_client,
    monkeypatch,
):
    app_module = sys.modules["src.api.app"]

    monkeypatch.setattr(
        app_module,
        "active_serving_bundle",
        None,
    )

    response = api_client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "No complete serving bundle is active."
    )

def test_readyz_reports_active_bundle(api_client):
    response = api_client.get("/readyz")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ready"
    assert body["serving_bundle_loaded"] is True
    assert body["model_version"] == "1"
    assert body["model_run_id"] == "test-run-1"

def test_predict_uses_active_bundle_instead_of_legacy_globals(
    api_client,
    api_headers,
    sample_prediction_payload,
    monkeypatch,
):
    app_module = sys.modules[
        "src.api.app"
    ]

    monkeypatch.setattr(
        app_module,
        "model",
        None,
    )
    monkeypatch.setattr(
        app_module,
        "store_metadata",
        None,
    )

    response = api_client.post(
        "/predict",
        json=sample_prediction_payload,
        headers=api_headers,
    )

    assert response.status_code == 200
    assert (
        response.json()["metadata"]["release_id"]
        == "release-test-v1"
    )

def test_rollback_validates_target_before_changing_pointer(
    api_client,
    api_headers,
    monkeypatch,
):

    app_module = sys.modules["src.api.app"]

    mock_pointer = MagicMock(
        return_value="release-current"
    )
    mock_load_target = MagicMock(
        side_effect=ValueError(
            "checksum mismatch"
        )
    )
    mock_activate_pointer = MagicMock()

    monkeypatch.setattr(
        app_module,
        "load_active_release_id",
        mock_pointer,
    )
    monkeypatch.setattr(
        app_module,
        "load_serving_bundle_for_release",
        mock_load_target,
    )
    monkeypatch.setattr(
        app_module,
        "activate_release_pointer",
        mock_activate_pointer,
    )

    response = api_client.post(
        "/admin/rollback-serving-release",
        headers=api_headers,
        json={
            "release_id": "release-broken",
        },
    )

    assert response.status_code == 500
    mock_activate_pointer.assert_not_called()

def test_rollback_activates_validated_release(
    api_client,
    api_headers,
    monkeypatch,
):

    app_module = sys.modules["src.api.app"]

    target_bundle = MagicMock()
    target_bundle.release_id = "release-old"
    target_bundle.model_version = "3"

    monkeypatch.setattr(
        app_module,
        "load_active_release_id",
        MagicMock(
            return_value="release-current"
        ),
    )
    monkeypatch.setattr(
        app_module,
        "load_serving_bundle_for_release",
        MagicMock(
            return_value=target_bundle
        ),
    )

    mock_pointer_update = MagicMock()
    mock_bundle_activation = MagicMock(
        return_value={
            "release_id": "release-old",
            "model_version": "3",
        }
    )

    monkeypatch.setattr(
        app_module,
        "activate_release_pointer",
        mock_pointer_update,
    )
    monkeypatch.setattr(
        app_module,
        "activate_serving_bundle",
        mock_bundle_activation,
    )

    response = api_client.post(
        "/admin/rollback-serving-release",
        headers=api_headers,
        json={
            "release_id": "release-old",
        },
    )

    assert response.status_code == 200

    mock_pointer_update.assert_called_once_with(
        models_path=app_module.MODELS_PATH,
        release_id="release-old",
        operation="rollback",
        previous_release_id=(
            "release-current"
        ),
    )

    mock_bundle_activation.assert_called_once_with(
        target_bundle
    )


def test_deployment_probe_is_not_logged(
    api_client,
    api_headers,
    sample_prediction_payload,
    mock_api_dependencies,
):
    payload = {
        **sample_prediction_payload,
        "context": {
            "purpose": (
                "post_deployment_verification"
            ),
        },
    }

    response = api_client.post(
        "/predict",
        json=payload,
        headers=api_headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["metadata"][
            "deployment_probe"
        ]
        is True
    )

    mock_api_dependencies.assert_not_called()