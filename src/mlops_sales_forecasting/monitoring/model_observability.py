import math
from collections.abc import Mapping
from dataclasses import dataclass

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)

from ..api.schema import PredictionResponse

FEATURE_DRIFT_SCORE = Gauge(
    "mlops_feature_drift_score",
    "Latest drift score reported for one monitored feature.",
    ["feature"],
)

FEATURE_DRIFT_DETECTED = Gauge(
    "mlops_feature_drift_detected",
    "Whether the latest feature drift score exceeds its threshold.",
    ["feature"],
)

FORECAST_VALUE = Histogram(
    "mlops_forecast_value",
    "Distribution of model forecast values.",
    buckets=(
        -1000000.0,
        -100000.0,
        -10000.0,
        -1000.0,
        -100.0,
        -10.0,
        0.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
    ),
)

FORECAST_HORIZON_STEP = Histogram(
    "mlops_forecast_horizon_step",
    "Distribution of requested forecast horizon steps.",
    buckets=(
        1.0,
        2.0,
        3.0,
        7.0,
        14.0,
        30.0,
        60.0,
        90.0,
        180.0,
        366.0,
    ),
)

FORECAST_FEEDBACK = Counter(
    "mlops_forecast_feedback_total",
    "Total number of forecasts with ground truth.",
)

FORECAST_ABSOLUTE_ERROR = Counter(
    "mlops_forecast_absolute_error_total",
    "Accumulated absolute forecast error.",
)

FORECAST_SQUARED_ERROR = Counter(
    "mlops_forecast_squared_error_total",
    "Accumulated squared forecast error.",
)

FORECAST_OVER_ERROR = Counter(
    "mlops_forecast_over_error_total",
    "Accumulated positive forecast error.",
)

FORECAST_UNDER_ERROR = Counter(
    "mlops_forecast_under_error_total",
    "Accumulated absolute negative forecast error.",
)


@dataclass(frozen=True, slots=True)
class ForecastFeedback:
    """Ground truth matched to one forecast value."""

    request_id: str
    row_index: int
    horizon_step: int
    prediction: float
    actual: float

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError(
                "request_id must not be empty"
            )

        if self.row_index < 0:
            raise ValueError(
                "row_index must not be negative"
            )

        if self.horizon_step < 1:
            raise ValueError(
                "horizon_step must be positive"
            )

        if not math.isfinite(self.prediction):
            raise ValueError(
                "prediction must be finite"
            )

        if not math.isfinite(self.actual):
            raise ValueError(
                "actual must be finite"
            )


def observe_model_outputs(
    response: PredictionResponse,
) -> None:
    """Record bounded forecasting output distributions."""
    for result in response.predictions:
        FORECAST_VALUE.observe(
            result.prediction
        )
        FORECAST_HORIZON_STEP.observe(
            result.horizon_step
        )


def observe_model_feedback(
    feedback: ForecastFeedback,
) -> None:
    """Record forecast performance after label arrival."""
    error = (
        feedback.prediction
        - feedback.actual
    )

    FORECAST_FEEDBACK.inc()
    FORECAST_ABSOLUTE_ERROR.inc(
        abs(error)
    )
    FORECAST_SQUARED_ERROR.inc(
        error**2
    )

    if error >= 0:
        FORECAST_OVER_ERROR.inc(error)
    else:
        FORECAST_UNDER_ERROR.inc(
            abs(error)
        )



def observe_feature_drift(
    *,
    scores: Mapping[str, float],
    threshold: float,
) -> None:
    """Publish project-computed feature drift scores."""
    if (
        not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError(
            "Drift threshold must be a finite, "
            "non-negative number."
        )

    for feature, score in scores.items():
        if not feature:
            raise ValueError(
                "Drift feature name must not be empty."
            )

        if (
            not math.isfinite(score)
            or score < 0
        ):
            raise ValueError(
                "Drift scores must be finite, "
                "non-negative numbers."
            )

        FEATURE_DRIFT_SCORE.labels(
            feature=feature,
        ).set(score)

        FEATURE_DRIFT_DETECTED.labels(
            feature=feature,
        ).set(
            float(score >= threshold)
        )