import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..data.contracts import DatasetSplits


def _validate_metrics(
    metrics: Mapping[str, float],
) -> None:
    """Validate a collection of numeric model metrics."""
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Metric names must be non-empty strings."
            )

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"Metric '{name}' must be numeric."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"Metric '{name}' must be finite."
            )


def _validate_artifacts(
    artifacts: Mapping[str, str],
) -> None:
    """Validate named training artifact locations."""
    for name, uri in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Artifact names must be non-empty strings."
            )

        if not isinstance(uri, str) or not uri:
            raise ValueError(
                f"Artifact URI for '{name}' "
                "must be a non-empty string."
            )


@dataclass(frozen=True)
class TrainingResult:
    """Result produced by a successful model-training step."""

    model: Any
    run_id: str
    metrics: Mapping[str, float]
    parameters: Mapping[str, Any] = field(
        default_factory=dict
    )
    artifacts: Mapping[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError(
                "Training result must contain a model."
            )

        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(
                "Training result must contain a run ID."
            )

        _validate_metrics(self.metrics)
        _validate_artifacts(self.artifacts)


@dataclass(frozen=True)
class EvaluationResult:
    """Model metrics and the resulting approval decision."""

    metrics: Mapping[str, float]
    approved: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_metrics(self.metrics)

        if not isinstance(self.approved, bool):
            raise TypeError(
                "Evaluation approval must be a boolean."
            )

        if not all(
            isinstance(reason, str) and reason
            for reason in self.reasons
        ):
            raise ValueError(
                "Evaluation reasons must be non-empty strings."
            )

        if not self.approved and not self.reasons:
            raise ValueError(
                "Rejected evaluations must contain a reason."
            )


@runtime_checkable
class ModelTrainer(Protocol):
    """Train a model from prepared dataset splits."""

    def train(
        self,
        datasets: DatasetSplits,
        config: Mapping[str, Any],
    ) -> TrainingResult:
        """Train and return a model candidate."""
        ...


@runtime_checkable
class ModelEvaluator(Protocol):
    """Evaluate a trained model candidate."""

    def evaluate(
        self,
        training_result: TrainingResult,
        datasets: DatasetSplits,
        config: Mapping[str, Any],
    ) -> EvaluationResult:
        """Evaluate whether a model candidate is acceptable."""
        ...