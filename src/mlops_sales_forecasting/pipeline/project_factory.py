from collections.abc import Mapping
from typing import Any

from mlops_sales_forecasting.data.splits.split import (
    RossmannDatasetSplitter,
)
from mlops_sales_forecasting.inference.releases.project_input_provider import (
    RossmannServingReleaseInputProvider,
)
from mlops_sales_forecasting.pipeline.factory import (
    build_training_pipeline,
)
from mlops_sales_forecasting.pipeline.project_adapters import (
    PersistingRossmannDataIngestor,
    PersistingRossmannFeatureBuilder,
)
from mlops_sales_forecasting.pipeline.service import (
    TrainingPipeline,
)
from mlops_sales_forecasting.training.evaluator import (
    RossmannModelEvaluator,
)
from mlops_sales_forecasting.training.model_logger import (
    XGBoostModelArtifactLogger,
)
from mlops_sales_forecasting.training.trainer import (
    RossmannModelTrainer,
)


def build_project_training_pipeline(
    config: Mapping[str, Any],
) -> TrainingPipeline:
    """Build the complete Rossmann forecasting pipeline."""
    training_config = config.get(
        "training",
        {},
    )
    is_drift_run = bool(
        training_config.get(
            "is_drift_run",
            False,
        )
    )

    return build_training_pipeline(
        ingestor=PersistingRossmannDataIngestor(),
        feature_builder=(PersistingRossmannFeatureBuilder()),
        splitter=RossmannDatasetSplitter(is_drift_run=is_drift_run),
        trainer=RossmannModelTrainer(is_drift_run=is_drift_run),
        evaluator=RossmannModelEvaluator(),
        model_logger=(XGBoostModelArtifactLogger()),
        release_input_provider=(RossmannServingReleaseInputProvider()),
        config=config,
    )
