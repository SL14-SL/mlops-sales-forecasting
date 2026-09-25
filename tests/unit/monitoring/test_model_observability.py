from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.api.schema import (
    PredictionResponse,
    PredictionResult,
)
from mlops_sales_forecasting.monitoring import (
    model_observability,
)
from mlops_sales_forecasting.monitoring.model_observability import (
    ForecastFeedback,
)


def test_observes_forecast_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = MagicMock()
    horizons = MagicMock()

    monkeypatch.setattr(
        model_observability,
        "FORECAST_VALUE",
        values,
    )
    monkeypatch.setattr(
        model_observability,
        "FORECAST_HORIZON_STEP",
        horizons,
    )

    response = PredictionResponse(
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

    model_observability.observe_model_outputs(
        response
    )

    assert values.observe.call_count == 2
    values.observe.assert_any_call(100.0)
    values.observe.assert_any_call(110.0)
    horizons.observe.assert_any_call(1)
    horizons.observe.assert_any_call(2)


def test_observes_forecast_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback_total = MagicMock()
    absolute_error = MagicMock()
    squared_error = MagicMock()
    over_error = MagicMock()
    under_error = MagicMock()

    monkeypatch.setattr(
        model_observability,
        "FORECAST_FEEDBACK",
        feedback_total,
    )
    monkeypatch.setattr(
        model_observability,
        "FORECAST_ABSOLUTE_ERROR",
        absolute_error,
    )
    monkeypatch.setattr(
        model_observability,
        "FORECAST_SQUARED_ERROR",
        squared_error,
    )
    monkeypatch.setattr(
        model_observability,
        "FORECAST_OVER_ERROR",
        over_error,
    )
    monkeypatch.setattr(
        model_observability,
        "FORECAST_UNDER_ERROR",
        under_error,
    )

    model_observability.observe_model_feedback(
        ForecastFeedback(
            request_id="request-1",
            row_index=0,
            horizon_step=1,
            prediction=110.0,
            actual=100.0,
        )
    )

    feedback_total.inc.assert_called_once_with()
    absolute_error.inc.assert_called_once_with(
        10.0
    )
    squared_error.inc.assert_called_once_with(
        100.0
    )
    over_error.inc.assert_called_once_with(
        10.0
    )
    under_error.inc.assert_not_called()


def test_rejects_invalid_forecast_feedback() -> None:
    with pytest.raises(
        ValueError,
        match="horizon_step",
    ):
        ForecastFeedback(
            request_id="request-1",
            row_index=0,
            horizon_step=0,
            prediction=100.0,
            actual=100.0,
        )



def test_observes_feature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = MagicMock()
    detected = MagicMock()

    monkeypatch.setattr(
        model_observability,
        "FEATURE_DRIFT_SCORE",
        scores,
    )
    monkeypatch.setattr(
        model_observability,
        "FEATURE_DRIFT_DETECTED",
        detected,
    )

    model_observability.observe_feature_drift(
        scores={
            "feature_a": 0.08,
            "feature_b": 0.31,
        },
        threshold=0.2,
    )

    scores.labels.assert_any_call(
        feature="feature_a"
    )
    scores.labels.assert_any_call(
        feature="feature_b"
    )

    detected.labels.return_value.set.assert_any_call(
        0.0
    )
    detected.labels.return_value.set.assert_any_call(
        1.0
    )


def test_rejects_invalid_drift_score() -> None:
    with pytest.raises(
        ValueError,
        match="Drift scores",
    ):
        model_observability.observe_feature_drift(
            scores={
                "feature_a": float("nan"),
            },
            threshold=0.2,
        )