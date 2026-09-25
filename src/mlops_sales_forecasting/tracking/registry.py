from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import mlflow

from ..training.contracts import (
    EvaluationResult,
    TrainingResult,
)


@dataclass(frozen=True)
class ModelRegistrationResult:
    """Result of an MLflow model-registration decision."""

    registered: bool
    run_id: str
    model_name: str
    model_version: str | None = None
    model_uri: str | None = None
    reason: str | None = None


def _model_name_from_config(
    config: Mapping[str, Any],
) -> str:
    """Return the configured registered-model name."""
    tracking = config.get("tracking")

    if not isinstance(tracking, Mapping):
        raise ValueError(
            "Config must contain a valid "
            "'tracking' section."
        )

    model_name = tracking.get("model_name")

    if (
        not isinstance(model_name, str)
        or not model_name.strip()
        or model_name.startswith("${")
    ):
        raise ValueError(
            "Tracking config 'model_name' must be "
            "a resolved non-empty string."
        )

    return model_name


def _validate_model_uri(
    model_uri: str,
) -> str:
    """Validate one MLflow model source URI."""

    if (
        not isinstance(model_uri, str)
        or not model_uri.strip()
        or model_uri.startswith("${")
    ):
        raise ValueError(
            "MLflow model URI must be a "
            "resolved non-empty string."
        )

    return model_uri


def _validate_artifact_path(
    artifact_path: str,
) -> str:
    """Validate an MLflow run-relative model path."""
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


def register_approved_model(
    *,
    training_result: TrainingResult,
    evaluation_result: EvaluationResult,
    config: Mapping[str, Any],
    artifact_path: str = "model",
    model_uri: str | None = None,
) -> ModelRegistrationResult:
    """Register an approved MLflow model candidate."""
    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "Model registration requires TrainingResult."
        )

    if not isinstance(
        evaluation_result,
        EvaluationResult,
    ):
        raise TypeError(
            "Model registration requires EvaluationResult."
        )

    model_name = _model_name_from_config(
        config
    )

    if not evaluation_result.approved:
        return ModelRegistrationResult(
            registered=False,
            run_id=training_result.run_id,
            model_name=model_name,
            reason="; ".join(
                evaluation_result.reasons
            ),
        )

    if model_uri is None:
        validated_artifact_path = (
            _validate_artifact_path(
                artifact_path
            )
        )
        registration_source_uri = (
            f"runs:/{training_result.run_id}/"
            f"{validated_artifact_path}"
        )
    else:
        registration_source_uri = (
            _validate_model_uri(
                model_uri
            )
        )

    model_version = mlflow.register_model(
        model_uri=registration_source_uri,
        name=model_name,
    )
    version = str(model_version.version)

    return ModelRegistrationResult(
        registered=True,
        run_id=training_result.run_id,
        model_name=model_name,
        model_version=version,
        model_uri=(
            f"models:/{model_name}/{version}"
        ),
    )