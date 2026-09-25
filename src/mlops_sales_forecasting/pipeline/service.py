from collections.abc import Mapping
from dataclasses import dataclass
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
from .runner import (
    TrackedPipelineResult,
    run_tracked_training_pipeline,
)


@dataclass(frozen=True)
class TrainingPipeline:
    """Configured framework-independent training pipeline."""

    ingestor: DataIngestor
    feature_builder: FeatureBuilder
    splitter: DatasetSplitter
    trainer: ModelTrainer
    evaluator: ModelEvaluator
    model_logger: ModelArtifactLogger
    release_input_provider: ServingReleaseInputProvider
    config: Mapping[str, Any]
    run_repository: PipelineRunRepository

    def run(
        self,
        *,
        run_id: str | None = None,
    ) -> TrackedPipelineResult:
        """Execute the configured training pipeline."""

        return run_tracked_training_pipeline(
            ingestor=self.ingestor,
            feature_builder=self.feature_builder,
            splitter=self.splitter,
            trainer=self.trainer,
            evaluator=self.evaluator,
            config=self.config,
            run_id=run_id,
            observer=self.run_repository.save,
        )