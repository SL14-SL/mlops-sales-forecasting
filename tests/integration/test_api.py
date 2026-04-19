import pandas as pd
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_api_dependencies(
    monkeypatch,
    mock_xgb_model,
    sample_store_metadata,
    sample_store_state,
):
    """
    Mock environment variables and app-level dependencies so API tests do not
    depend on MLflow, GCS, startup loading, or the full feature pipeline.
    """
    monkeypatch.setenv("API_KEY", "test-secret-key")
    monkeypatch.setenv("APP_ENV", "dev")

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

    with (
        patch("src.api.app.model", mock_xgb_model),
        patch("src.api.app.store_metadata", sample_store_metadata),
        patch("src.api.app.store_state", sample_store_state),
        patch("src.api.app.model_type", "xgboost"),
        patch("src.api.app.target_transformation", "log1p"),
        patch("src.api.app.preprocess_data", return_value=mocked_processed_df),
        patch("src.api.app.align_features_for_model", return_value=mocked_processed_df),
        patch("src.api.app.log_prediction"),
    ):
        yield


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