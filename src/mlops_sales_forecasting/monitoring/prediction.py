from typing import Literal

from prometheus_client import Counter, Histogram

PredictionStatus = Literal[
    "success",
    "error",
]

SUPPORTED_TASK_TYPES = {
    "classification",
    "forecasting",
}

PREDICTION_REQUESTS = Counter(
    "mlops_prediction_requests_total",
    "Total number of model prediction requests.",
    [
        "task_type",
        "status",
    ],
)

PREDICTION_OBSERVATIONS = Counter(
    "mlops_prediction_observations_total",
    "Total number of observations submitted for prediction.",
    [
        "task_type",
    ],
)

PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds",
    "Model prediction latency in seconds.",
    [
        "task_type",
    ],
)


def observe_prediction(
    *,
    task_type: str,
    status: PredictionStatus,
    observation_count: int,
    latency_seconds: float,
) -> None:
    """Record bounded operational prediction metrics."""
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            "Unsupported prediction task type: "
            f"{task_type!r}."
        )

    if status not in {
        "success",
        "error",
    }:
        raise ValueError(
            "Prediction status must be "
            "'success' or 'error'."
        )

    if observation_count < 0:
        raise ValueError(
            "Prediction observation count "
            "must not be negative."
        )

    if latency_seconds < 0:
        raise ValueError(
            "Prediction latency must not "
            "be negative."
        )

    PREDICTION_REQUESTS.labels(
        task_type=task_type,
        status=status,
    ).inc()

    if observation_count:
        PREDICTION_OBSERVATIONS.labels(
            task_type=task_type,
        ).inc(observation_count)

    PREDICTION_LATENCY.labels(
        task_type=task_type,
    ).observe(latency_seconds)