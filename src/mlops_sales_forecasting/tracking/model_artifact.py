from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from ..training.contracts import TrainingResult


@runtime_checkable
class ModelArtifactLogger(Protocol):
    """Log a trained model to the active MLflow run."""

    def log_model(
        self,
        training_result: TrainingResult,
        *,
        artifact_path: str,
        config: Mapping[str, Any],
    ) -> str:
        """Log the model and return its MLflow model URI."""
        ...


@dataclass(frozen=True)
class LoggedModelArtifact:
    """Reference to a model logged in an MLflow run."""

    run_id: str
    artifact_path: str
    model_uri: str


def _validate_artifact_path(
    artifact_path: str,
) -> str:
    """Validate an MLflow run-relative artifact path."""

    if (
        not isinstance(artifact_path, str)
        or not artifact_path.strip()
    ):
        raise ValueError(
            "Model artifact path must not be empty."
        )

    path = PurePosixPath(artifact_path)

    if (
        path.is_absolute()
        or ".." in path.parts
        or artifact_path.startswith("/")
    ):
        raise ValueError(
            "Model artifact path must be relative."
        )

    return str(path)


def log_model_artifact(
    *,
    logger: ModelArtifactLogger,
    training_result: TrainingResult,
    config: Mapping[str, Any],
    artifact_path: str = "model",
) -> LoggedModelArtifact:
    """Log a model through its project-specific adapter."""

    if not isinstance(
        logger,
        ModelArtifactLogger,
    ):
        raise TypeError(
            "Model artifact logger must implement "
            "ModelArtifactLogger."
        )

    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "Model artifact logging requires "
            "TrainingResult."
        )

    validated_path = _validate_artifact_path(
        artifact_path
    )

    model_uri = logger.log_model(
        training_result,
        artifact_path=validated_path,
        config=config,
    )

    if (
        not isinstance(model_uri, str)
        or not model_uri.strip()
    ):
        raise ValueError(
            "Model artifact logger must return "
            "a non-empty MLflow model URI."
        )

    return LoggedModelArtifact(
        run_id=training_result.run_id,
        artifact_path=validated_path,
        model_uri=model_uri,
    )