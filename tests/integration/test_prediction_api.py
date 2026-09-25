from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from mlops_sales_forecasting.api.app import create_app
from mlops_sales_forecasting.api.routers import prediction
from mlops_sales_forecasting.api.schema import (
    PredictionResponse,
    PredictionResult,
)
from mlops_sales_forecasting.inference.model_manager import (
    ModelNotReadyError,
)

API_KEY = "test-prediction-key"


def build_client(
    *,
    ready: bool = True,
) -> tuple[TestClient, MagicMock]:
    model_manager = MagicMock()
    model_manager.ready = ready

    bundle = MagicMock()
    bundle.release_id = "release-1"
    bundle.model_name = "prediction-test-model"
    bundle.model_version = "1"
    bundle.manifest.task_type.value = (
        "forecasting"
    )
    model_manager.get_bundle.return_value = bundle

    app = create_app(
        model_manager=model_manager,
        load_model_on_startup=False,
        api_key=API_KEY,
    )

    return TestClient(
        app,
        raise_server_exceptions=False,
    ), model_manager



def valid_payload() -> dict:
    return {
        "inputs": [
            {
                "entity_id": 1,
                "date": "2026-09-16",
            }
        ],
        "horizon": 2,
    }


def successful_response() -> PredictionResponse:
    return PredictionResponse(
        release_id="release-1",
        predictions=[
            PredictionResult(
                row_index=0,
                horizon_step=1,
                prediction=100.0,
            ),
            PredictionResult(
                row_index=0,
                horizon_step=2,
                prediction=110.0,
            ),
        ],
    )



def test_predict_returns_success(
    monkeypatch,
) -> None:
    client, model_manager = build_client()
    expected_response = successful_response()
    service = MagicMock()
    service.predict.return_value = (
        expected_response
    )
    service_constructor = MagicMock(
        return_value=service
    )
    observe = MagicMock()
    observe_outputs = MagicMock()

    monkeypatch.setattr(
        prediction,
        "PredictionService",
        service_constructor,
    )
    monkeypatch.setattr(
        prediction,
        "observe_prediction",
        observe,
    )
    monkeypatch.setattr(
        prediction,
        "observe_model_outputs",
        observe_outputs,
    )

    with client:
        response = client.post(
            "/predict",
            headers={
                "X-API-Key": API_KEY,
            },
            json=valid_payload(),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["release_id"] == "release-1"
    service_constructor.assert_called_once_with(
        model_manager
    )
    observe_outputs.assert_called_once_with(
        expected_response
    )

    observe.assert_called_once()

    metric_call = observe.call_args.kwargs

    assert metric_call["task_type"] == (
        "forecasting"
    )
    assert metric_call["status"] == "success"
    assert metric_call["observation_count"] == 1
    assert metric_call["latency_seconds"] >= 0


def test_predict_requires_api_key() -> None:
    client, _ = build_client()

    with client:
        response = client.post(
            "/predict",
            json=valid_payload(),
        )

    assert response.status_code == 401


def test_predict_rejects_invalid_api_key() -> None:
    client, _ = build_client()

    with client:
        response = client.post(
            "/predict",
            headers={
                "X-API-Key": "wrong-key",
            },
            json=valid_payload(),
        )

    assert response.status_code == 403


def test_predict_requires_ready_model_manager() -> None:
    client, _ = build_client(ready=False)

    with client:
        response = client.post(
            "/predict",
            headers={
                "X-API-Key": API_KEY,
            },
            json=valid_payload(),
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction service is not ready."
    )


def test_predict_rejects_empty_input_batch() -> None:
    client, _ = build_client()

    with client:
        response = client.post(
            "/predict",
            headers={
                "X-API-Key": API_KEY,
            },
            json={
                "inputs": [],
            },
        )

    assert response.status_code == 422


def test_predict_maps_domain_validation_error(
    monkeypatch,
) -> None:
    client, _ = build_client()
    service = MagicMock()
    service.predict.side_effect = ValueError(
        "Prediction input is missing required columns."
    )
    observe = MagicMock()
    observe_outputs = MagicMock()

    monkeypatch.setattr(
        prediction,
        "PredictionService",
        MagicMock(return_value=service),
    )
    monkeypatch.setattr(
        prediction,
        "observe_prediction",
        observe,
    )
    monkeypatch.setattr(
        prediction,
        "observe_model_outputs",
        observe_outputs,
    )

    with client:
        response = client.post(
            "/predict",
            headers={
                "X-API-Key": API_KEY,
            },
            json=valid_payload(),
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Prediction input is missing required columns."
    )

    observe_outputs.assert_not_called()
    observe.assert_called_once()

    metric_call = observe.call_args.kwargs

    assert metric_call["task_type"] == (
        "forecasting"
    )
    assert metric_call["status"] == "error"
    assert metric_call["observation_count"] == 1
    assert metric_call["latency_seconds"] >= 0


def test_predict_maps_model_not_ready_error(
    monkeypatch,
) -> None:
    client, _ = build_client()
    service = MagicMock()
    service.predict.side_effect = ModelNotReadyError(
        "Bundle disappeared during reload."
    )
    monkeypatch.setattr(
        prediction,
        "PredictionService",
        MagicMock(return_value=service),
    )

    with client:
        response = client.post(
            "/predict",
            headers={
                "X-API-Key": API_KEY,
            },
            json=valid_payload(),
        )

    assert response.status_code == 503


def test_predict_hides_unexpected_error(
    monkeypatch,
) -> None:
    client, _ = build_client()
    service = MagicMock()
    service.predict.side_effect = RuntimeError(
        "sensitive model backend details"
    )
    monkeypatch.setattr(
        prediction,
        "PredictionService",
        MagicMock(return_value=service),
    )

    with client:
        response = client.post(
            "/predict",
            headers={
                "X-API-Key": API_KEY,
                "X-Request-ID": "prediction-error-1",
            },
            json=valid_payload(),
        )

    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "error": "internal_server_error",
        "request_id": "prediction-error-1",
    }
    assert (
        "sensitive model backend details"
        not in response.text
    )