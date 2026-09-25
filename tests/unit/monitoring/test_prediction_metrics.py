from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.monitoring import (
    prediction,
)


def install_metric_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    MagicMock,
    MagicMock,
    MagicMock,
]:
    requests = MagicMock()
    observations = MagicMock()
    latency = MagicMock()

    monkeypatch.setattr(
        prediction,
        "PREDICTION_REQUESTS",
        requests,
    )
    monkeypatch.setattr(
        prediction,
        "PREDICTION_OBSERVATIONS",
        observations,
    )
    monkeypatch.setattr(
        prediction,
        "PREDICTION_LATENCY",
        latency,
    )

    return (
        requests,
        observations,
        latency,
    )


def test_observes_successful_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        requests,
        observations,
        latency,
    ) = install_metric_mocks(monkeypatch)

    prediction.observe_prediction(
        task_type="forecasting",
        status="success",
        observation_count=3,
        latency_seconds=0.125,
    )

    requests.labels.assert_called_once_with(
        task_type="forecasting",
        status="success",
    )
    requests.labels.return_value.inc.assert_called_once_with()

    observations.labels.assert_called_once_with(
        task_type="forecasting",
    )
    observations.labels.return_value.inc.assert_called_once_with(
        3
    )

    latency.labels.assert_called_once_with(
        task_type="forecasting",
    )
    latency.labels.return_value.observe.assert_called_once_with(
        0.125
    )


def test_observes_failed_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        requests,
        observations,
        latency,
    ) = install_metric_mocks(monkeypatch)

    prediction.observe_prediction(
        task_type="forecasting",
        status="error",
        observation_count=0,
        latency_seconds=0.05,
    )

    requests.labels.assert_called_once_with(
        task_type="forecasting",
        status="error",
    )
    observations.labels.assert_not_called()
    latency.labels.return_value.observe.assert_called_once_with(
        0.05
    )


@pytest.mark.parametrize(
    "task_type",
    [
        "",
        "regression",
        "unknown",
    ],
)
def test_rejects_unsupported_task_type(
    task_type: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported prediction task type",
    ):
        prediction.observe_prediction(
            task_type=task_type,
            status="success",
            observation_count=1,
            latency_seconds=0.1,
        )


def test_rejects_invalid_status() -> None:
    with pytest.raises(
        ValueError,
        match="Prediction status",
    ):
        prediction.observe_prediction(
            task_type="forecasting",
            status="invalid",  # type: ignore[arg-type]
            observation_count=1,
            latency_seconds=0.1,
        )


def test_rejects_negative_observation_count() -> None:
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        prediction.observe_prediction(
            task_type="forecasting",
            status="success",
            observation_count=-1,
            latency_seconds=0.1,
        )


def test_rejects_negative_latency() -> None:
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        prediction.observe_prediction(
            task_type="forecasting",
            status="success",
            observation_count=1,
            latency_seconds=-0.1,
        )