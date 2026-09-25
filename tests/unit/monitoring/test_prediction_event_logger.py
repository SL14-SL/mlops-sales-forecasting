from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from mlops_sales_forecasting.api.routers import prediction
from mlops_sales_forecasting.monitoring.prediction_event_logger import (
    PredictionEvent,
)

APPLICATION_CONFIG = {
    "data": {
        "target_column": "Sales",
        "time_column": "Date",
        "id_columns": [
            "Store",
        ],
    },
}

def build_http_request() -> MagicMock:
    request = MagicMock()
    request.state = SimpleNamespace(
        request_id="request-123"
    )
    return request


def build_prediction_request(
    batch_size: int = 2,
) -> MagicMock:
    request = MagicMock()
    request.inputs = [
        {"feature": index}
        for index in range(batch_size)
    ]
    return request


def build_model_manager() -> tuple[MagicMock, MagicMock]:
    task_type = MagicMock()
    task_type.value = "forecasting"

    manifest = MagicMock()
    manifest.task_type = task_type

    bundle = MagicMock()
    bundle.release_id = "release-456"
    bundle.model_name = "example-model"
    bundle.model_version = "7"
    bundle.manifest = manifest

    manager = MagicMock()
    manager.ready = True
    manager.get_bundle.return_value = bundle

    return manager, bundle


def test_successful_prediction_records_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = build_model_manager()
    http_request = build_http_request()
    prediction_request = build_prediction_request(
        batch_size=3
    )
    response = MagicMock()

    service = MagicMock()
    service.predict.return_value = response

    service_factory = MagicMock(
        return_value=service
    )
    event_logger = MagicMock()

    monkeypatch.setattr(
        prediction,
        "PredictionService",
        service_factory,
    )
    monkeypatch.setattr(
        prediction,
        "log_prediction_event",
        event_logger,
    )
    monkeypatch.setattr(
        prediction.time,
        "perf_counter",
        MagicMock(
            side_effect=[10.0, 10.025]
        ),
    )

    result = prediction.predict(
        http_request=http_request,
        request=prediction_request,
        _=None,
        model_manager=manager,
        application_config=APPLICATION_CONFIG,
    )

    assert result is response
    service_factory.assert_called_once_with(manager, APPLICATION_CONFIG)
    service.predict.assert_called_once_with(
        prediction_request
    )
    event_logger.assert_called_once()

    event = event_logger.call_args.args[0]

    assert isinstance(event, PredictionEvent)
    assert event.request_id == "request-123"
    assert event.release_id == "release-456"
    assert event.task_type == "forecasting"
    assert event.model_name == "example-model"
    assert event.model_version == "7"
    assert event.batch_size == 3
    assert event.duration_ms == pytest.approx(25.0)
    assert event.status == "success"
    assert event.error_type is None


def test_validation_error_records_failed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = build_model_manager()
    http_request = build_http_request()
    prediction_request = build_prediction_request()

    service = MagicMock()
    service.predict.side_effect = ValueError(
        "Invalid prediction input."
    )

    event_logger = MagicMock()

    monkeypatch.setattr(
        prediction,
        "PredictionService",
        MagicMock(return_value=service),
    )
    monkeypatch.setattr(
        prediction,
        "log_prediction_event",
        event_logger,
    )
    monkeypatch.setattr(
        prediction.time,
        "perf_counter",
        MagicMock(
            side_effect=[20.0, 20.01]
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        prediction.predict(
            http_request=http_request,
            request=prediction_request,
            _=None,
            model_manager=manager,
            application_config=APPLICATION_CONFIG,
        )

    assert exc_info.value.status_code == 422

    event = event_logger.call_args.args[0]

    assert event.status == "error"
    assert event.error_type == "ValueError"
    assert event.duration_ms == pytest.approx(10.0)


def test_model_not_ready_error_records_failed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = build_model_manager()
    http_request = build_http_request()
    prediction_request = build_prediction_request()

    service = MagicMock()
    service.predict.side_effect = (
        prediction.ModelNotReadyError(
            "Bundle unavailable."
        )
    )

    event_logger = MagicMock()

    monkeypatch.setattr(
        prediction,
        "PredictionService",
        MagicMock(return_value=service),
    )
    monkeypatch.setattr(
        prediction,
        "log_prediction_event",
        event_logger,
    )

    with pytest.raises(HTTPException) as exc_info:
        prediction.predict(
            http_request=http_request,
            request=prediction_request,
            _=None,
            model_manager=manager,
            application_config=APPLICATION_CONFIG,
        )

    assert exc_info.value.status_code == 503

    event = event_logger.call_args.args[0]

    assert event.status == "error"
    assert event.error_type == "ModelNotReadyError"


def test_unexpected_error_is_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = build_model_manager()
    http_request = build_http_request()
    prediction_request = build_prediction_request()

    service = MagicMock()
    service.predict.side_effect = RuntimeError(
        "Internal model failure."
    )

    event_logger = MagicMock()

    monkeypatch.setattr(
        prediction,
        "PredictionService",
        MagicMock(return_value=service),
    )
    monkeypatch.setattr(
        prediction,
        "log_prediction_event",
        event_logger,
    )

    with pytest.raises(
        RuntimeError,
        match="Internal model failure",
    ):
        prediction.predict(
            http_request=http_request,
            request=prediction_request,
            _=None,
            model_manager=manager,
            application_config=APPLICATION_CONFIG,
        )

    event = event_logger.call_args.args[0]

    assert event.status == "error"
    assert event.error_type == "RuntimeError"


def test_request_without_loaded_bundle_is_not_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MagicMock()
    manager.ready = False
    event_logger = MagicMock()

    monkeypatch.setattr(
        prediction,
        "log_prediction_event",
        event_logger,
    )

    with pytest.raises(HTTPException) as exc_info:
        prediction.predict(
            http_request=build_http_request(),
            request=build_prediction_request(),
            _=None,
            model_manager=manager,
            application_config=APPLICATION_CONFIG,
        )

    assert exc_info.value.status_code == 503
    event_logger.assert_not_called()

def test_logging_failure_does_not_break_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = build_model_manager()
    response = MagicMock()

    service = MagicMock()
    service.predict.return_value = response

    monkeypatch.setattr(
        prediction,
        "PredictionService",
        MagicMock(return_value=service),
    )
    monkeypatch.setattr(
        prediction,
        "log_prediction_event",
        MagicMock(
            side_effect=TypeError(
                "Event is not serializable."
            )
        ),
    )

    result = prediction.predict(
        http_request=build_http_request(),
        request=build_prediction_request(),
        _=None,
        model_manager=manager,
        application_config=APPLICATION_CONFIG,
    )

    assert result is response