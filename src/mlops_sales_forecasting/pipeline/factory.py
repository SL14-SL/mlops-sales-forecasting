from collections.abc import Mapping
from typing import Any

from ..data.contracts import (
    DataIngestor,
    DatasetSplitter,
    FeatureBuilder,
)
from ..inference.releases.input_provider import (
    ServingReleaseInputProvider,
)
from ..tracking.model_artifact import ModelArtifactLogger
from ..training.contracts import (
    ModelEvaluator,
    ModelTrainer,
)
from .repository import PipelineRunRepository
from .service import TrainingPipeline


def _require_paths(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the validated paths configuration."""
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError(
            "Config must contain a valid 'paths' section."
        )

    return paths


def _require_path(
    paths: Mapping[str, Any],
    name: str,
) -> str:
    """Return one resolved path configuration value."""
    value = paths.get(name)

    if (
        not isinstance(value, str)
        or not value.strip()
        or value.startswith("${")
    ):
        raise ValueError(
            f"Config path '{name}' must be "
            "a resolved non-empty string."
        )

    return value.rstrip("/")


def build_pipeline_run_repository(
    config: Mapping[str, Any],
) -> PipelineRunRepository:
    """Build the configured pipeline lifecycle repository."""
    paths = _require_paths(config)
    configured_path = paths.get(
        "pipeline_runs"
    )

    if configured_path is not None:
        root_path = _require_path(
            paths,
            "pipeline_runs",
        )
    else:
        artifacts_path = _require_path(
            paths,
            "artifacts",
        )
        root_path = (
            f"{artifacts_path}/pipeline-runs"
        )

    return PipelineRunRepository(root_path)


def build_training_pipeline(
    *,
    ingestor: DataIngestor,
    feature_builder: FeatureBuilder,
    splitter: DatasetSplitter,
    trainer: ModelTrainer,
    evaluator: ModelEvaluator,
    model_logger: ModelArtifactLogger,
    release_input_provider: (
        ServingReleaseInputProvider
    ),
    config: Mapping[str, Any],
) -> TrainingPipeline:
    """Build a training pipeline with lifecycle persistence."""

    return TrainingPipeline(
        ingestor=ingestor,
        feature_builder=feature_builder,
        splitter=splitter,
        trainer=trainer,
        evaluator=evaluator,
        model_logger=model_logger,
        release_input_provider=(
            release_input_provider
        ),
        config=config,
        run_repository=(
            build_pipeline_run_repository(
                config
            )
        ),
    )