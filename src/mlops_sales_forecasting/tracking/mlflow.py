from collections.abc import (
    Iterator,
    Mapping,
)
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlflow

from ..training.contracts import EvaluationResult, TrainingResult


@dataclass(frozen=True)
class MlflowTrackingSettings:
    """Validated MLflow training configuration."""

    tracking_uri: str
    experiment_name: str


def _require_string(
    section: Mapping[str, Any],
    name: str,
) -> str:
    """Return one resolved non-empty configuration value."""
    value = section.get(name)

    if (
        not isinstance(value, str)
        or not value.strip()
        or value.startswith("${")
    ):
        raise ValueError(
            f"Tracking config '{name}' must be "
            "a resolved non-empty string."
        )

    return value


def load_mlflow_tracking_settings(
    config: Mapping[str, Any],
) -> MlflowTrackingSettings:
    """Load MLflow settings from application configuration."""
    tracking = config.get("tracking")

    if not isinstance(tracking, Mapping):
        raise ValueError(
            "Config must contain a valid "
            "'tracking' section."
        )

    return MlflowTrackingSettings(
        tracking_uri=_require_string(
            tracking,
            "mlflow_tracking_uri",
        ),
        experiment_name=_require_string(
            tracking,
            "experiment_name",
        ),
    )


def configure_mlflow_tracking(
    settings: MlflowTrackingSettings,
) -> None:
    """Configure MLflow tracking and experiment selection."""
    mlflow.set_tracking_uri(
        settings.tracking_uri
    )
    mlflow.set_experiment(
        settings.experiment_name
    )


@contextmanager
def start_training_run(
    config: Mapping[str, Any],
    *,
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[str]:
    """Start an MLflow training run and yield its run ID."""
    settings = load_mlflow_tracking_settings(
        config
    )
    configure_mlflow_tracking(settings)

    with mlflow.start_run(
        run_name=run_name,
        tags=(
            dict(tags)
            if tags is not None
            else None
        ),
    ) as active_run:
        yield active_run.info.run_id


def _active_run_id() -> str:
    """Return the active MLflow run ID."""
    active_run = mlflow.active_run()

    if active_run is None:
        raise RuntimeError(
            "Training result logging requires "
            "an active MLflow run."
        )

    return active_run.info.run_id

def get_active_training_run_id() -> str:
    """Return the active MLflow training-run ID."""

    return _active_run_id()

def log_training_result(
    result: TrainingResult,
) -> TrainingResult:
    """Log a training result to its active MLflow run."""
    if not isinstance(result, TrainingResult):
        raise TypeError(
            "MLflow tracking requires TrainingResult."
        )

    active_run_id = _active_run_id()

    if result.run_id != active_run_id:
        raise ValueError(
            "Training result run ID does not match "
            "the active MLflow run."
        )

    if result.parameters:
        mlflow.log_params(
            dict(result.parameters)
        )

    if result.metrics:
        mlflow.log_metrics(
            {
                name: float(value)
                for name, value
                in result.metrics.items()
            }
        )

    if result.artifacts:
        mlflow.set_tags(
            {
                f"artifact_uri.{name}": uri
                for name, uri
                in result.artifacts.items()
            }
        )

    return result

def log_evaluation_result(
    result: EvaluationResult,
) -> EvaluationResult:
    """Log evaluation metrics and approval to the active run."""

    if not isinstance(result, EvaluationResult):
        raise TypeError(
            "MLflow evaluation logging requires "
            "EvaluationResult."
        )

    _active_run_id()

    if result.metrics:
        mlflow.log_metrics(
            {
                name: float(value)
                for name, value
                in result.metrics.items()
            }
        )

    tags = {
        "evaluation.approved": (
            "true"
            if result.approved
            else "false"
        )
    }

    if result.reasons:
        tags["evaluation.reasons"] = "; ".join(
            result.reasons
        )

    mlflow.set_tags(tags)

    return result