from collections.abc import Mapping
from typing import Any

from .service import TrainingPipeline


def build_project_training_pipeline(
    config: Mapping[str, Any],
) -> TrainingPipeline:
    """Build the project-specific forecasting pipeline."""
    del config

    raise NotImplementedError(
        "Implement the project-specific forecasting "
        "training pipeline by providing a DataIngestor, "
        "FeatureBuilder, DatasetSplitter, ModelTrainer, "
        "ModelEvaluator, ModelArtifactLogger and "
        "ServingReleaseInputProvider. The ModelTrainer "
        "must use get_active_training_run_id() for the "
        "TrainingResult run_id."
    )