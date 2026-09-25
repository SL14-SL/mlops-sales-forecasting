from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...training.contracts import (
    EvaluationResult,
    TrainingResult,
)
from .artifact_publisher import (
    ServingArtifactSource,
)
from .contracts import TaskType


@dataclass(frozen=True)
class ServingReleaseInput:
    """Project-specific input required to publish a release."""

    task_type: TaskType
    model_type: str
    sources: Mapping[
        str,
        ServingArtifactSource,
    ]
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    dataset_version: str | None = None
    config_hash: str | None = None
    git_commit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.task_type,
            TaskType,
        ):
            raise TypeError(
                "Serving release input requires TaskType."
            )

        if (
            not isinstance(self.model_type, str)
            or not self.model_type.strip()
        ):
            raise ValueError(
                "Serving release model type "
                "must not be empty."
            )

        if not isinstance(
            self.sources,
            Mapping,
        ):
            raise TypeError(
                "Serving release sources must be a mapping."
            )

        if not self.sources:
            raise ValueError(
                "Serving release input requires "
                "at least one artifact source."
            )

        for name, source in self.sources.items():
            if (
                not isinstance(name, str)
                or not name.strip()
            ):
                raise ValueError(
                    "Serving artifact name "
                    "must not be empty."
                )

            if not isinstance(
                source,
                ServingArtifactSource,
            ):
                raise TypeError(
                    "Serving release source must be "
                    "ServingArtifactSource."
                )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise TypeError(
                "Serving release metadata "
                "must be a mapping."
            )


@runtime_checkable
class ServingReleaseInputProvider(Protocol):
    """Build task-specific inputs for a serving release."""

    def build_release_input(
        self,
        *,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
        config: Mapping[str, Any],
    ) -> ServingReleaseInput:
        """Return artifacts and metadata for one release."""
        ...


def build_serving_release_input(
    *,
    provider: ServingReleaseInputProvider,
    training_result: TrainingResult,
    evaluation_result: EvaluationResult,
    config: Mapping[str, Any],
) -> ServingReleaseInput:
    """Build and validate project-specific release input."""

    if not isinstance(
        provider,
        ServingReleaseInputProvider,
    ):
        raise TypeError(
            "Release input provider must implement "
            "ServingReleaseInputProvider."
        )

    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "Release input creation requires "
            "TrainingResult."
        )

    if not isinstance(
        evaluation_result,
        EvaluationResult,
    ):
        raise TypeError(
            "Release input creation requires "
            "EvaluationResult."
        )

    release_input = (
        provider.build_release_input(
            training_result=training_result,
            evaluation_result=(
                evaluation_result
            ),
            config=config,
        )
    )

    if not isinstance(
        release_input,
        ServingReleaseInput,
    ):
        raise TypeError(
            "Release input provider must return "
            "ServingReleaseInput."
        )

    return release_input