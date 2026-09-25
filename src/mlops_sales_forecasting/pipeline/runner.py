from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from ..data.contracts import (
    DataIngestor,
    DatasetCollection,
    DatasetSplits,
    DatasetSplitter,
    FeatureBuilder,
)
from ..training.contracts import (
    EvaluationResult,
    ModelEvaluator,
    ModelTrainer,
    TrainingResult,
)
from .status import PipelineRun


@dataclass(frozen=True)
class PipelineResult:
    """Complete result of one training-pipeline execution."""

    datasets: DatasetCollection
    features: pd.DataFrame
    splits: DatasetSplits
    training: TrainingResult
    evaluation: EvaluationResult


@dataclass(frozen=True)
class TrackedPipelineResult:
    """Pipeline result together with its final lifecycle state."""

    pipeline: PipelineResult
    run: PipelineRun


PipelineRunObserver = Callable[
    [PipelineRun],
    None,
]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _ignore_pipeline_status(
    run: PipelineRun,
) -> None:
    """Default observer used when status persistence is disabled."""
    del run


def _validate_features(
    features: object,
) -> pd.DataFrame:
    """Validate the result of feature engineering."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "Feature builder must return a pandas DataFrame."
        )

    if features.empty:
        raise ValueError(
            "Feature builder returned an empty DataFrame."
        )

    return features


def _validate_splits(
    splits: object,
) -> DatasetSplits:
    """Validate the result of dataset splitting."""
    if not isinstance(splits, DatasetSplits):
        raise TypeError(
            "Dataset splitter must return DatasetSplits."
        )

    return splits


def _validate_training_result(
    result: object,
) -> TrainingResult:
    """Validate the result of model training."""
    if not isinstance(result, TrainingResult):
        raise TypeError(
            "Model trainer must return TrainingResult."
        )

    return result


def _validate_evaluation_result(
    result: object,
) -> EvaluationResult:
    """Validate the result of model evaluation."""
    if not isinstance(result, EvaluationResult):
        raise TypeError(
            "Model evaluator must return EvaluationResult."
        )

    return result


def run_training_pipeline(
    *,
    ingestor: DataIngestor,
    feature_builder: FeatureBuilder,
    splitter: DatasetSplitter,
    trainer: ModelTrainer,
    evaluator: ModelEvaluator,
    config: Mapping[str, Any],
) -> PipelineResult:
    """Execute the framework-independent training pipeline."""
    datasets = ingestor.ingest(config)

    if not isinstance(datasets, DatasetCollection):
        raise TypeError(
            "Data ingestor must return DatasetCollection."
        )

    features = _validate_features(
        feature_builder.build_features(
            datasets,
            config,
        )
    )
    splits = _validate_splits(
        splitter.split(
            features,
            config,
        )
    )
    training_result = _validate_training_result(
        trainer.train(
            splits,
            config,
        )
    )
    evaluation_result = _validate_evaluation_result(
        evaluator.evaluate(
            training_result,
            splits,
            config,
        )
    )

    return PipelineResult(
        datasets=datasets,
        features=features,
        splits=splits,
        training=training_result,
        evaluation=evaluation_result,
    )


def run_tracked_training_pipeline(
    *,
    ingestor: DataIngestor,
    feature_builder: FeatureBuilder,
    splitter: DatasetSplitter,
    trainer: ModelTrainer,
    evaluator: ModelEvaluator,
    config: Mapping[str, Any],
    run_id: str | None = None,
    observer: PipelineRunObserver = (
        _ignore_pipeline_status
    ),
    clock: Clock = _utc_now,
) -> TrackedPipelineResult:
    """Execute a pipeline and report lifecycle transitions."""
    running = PipelineRun.start(
        run_id or str(uuid4()),
        started_at_utc=clock(),
    )
    observer(running)

    try:
        pipeline_result = run_training_pipeline(
            ingestor=ingestor,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            config=config,
        )
    except Exception as exc:
        failed = running.fail(
            exc,
            completed_at_utc=clock(),
        )
        observer(failed)
        raise

    if pipeline_result.evaluation.approved:
        finished = running.succeed(
            completed_at_utc=clock(),
        )
    else:
        finished = running.reject(
            pipeline_result.evaluation.reasons,
            completed_at_utc=clock(),
        )

    observer(finished)

    return TrackedPipelineResult(
        pipeline=pipeline_result,
        run=finished,
    )