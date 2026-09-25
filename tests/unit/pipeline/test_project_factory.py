from mlops_sales_forecasting.data.splits.split import (
    RossmannDatasetSplitter,
)
from mlops_sales_forecasting.inference.releases.project_input_provider import (
    RossmannServingReleaseInputProvider,
)
from mlops_sales_forecasting.pipeline.project_adapters import (
    PersistingRossmannDataIngestor,
    PersistingRossmannFeatureBuilder,
)
from mlops_sales_forecasting.pipeline.project_factory import (
    build_project_training_pipeline,
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


def test_build_project_training_pipeline(
    tmp_path,
) -> None:
    config = {
        "paths": {
            "artifacts": str(tmp_path / "artifacts"),
        },
        "training": {
            "is_drift_run": False,
        },
    }

    result = build_project_training_pipeline(config)

    assert isinstance(
        result,
        TrainingPipeline,
    )
    assert isinstance(
        result.ingestor,
        PersistingRossmannDataIngestor,
    )
    assert isinstance(
        result.feature_builder,
        PersistingRossmannFeatureBuilder,
    )
    assert isinstance(
        result.splitter,
        RossmannDatasetSplitter,
    )
    assert isinstance(
        result.trainer,
        RossmannModelTrainer,
    )
    assert isinstance(
        result.evaluator,
        RossmannModelEvaluator,
    )
    assert isinstance(
        result.model_logger,
        XGBoostModelArtifactLogger,
    )
    assert isinstance(
        result.release_input_provider,
        RossmannServingReleaseInputProvider,
    )
    assert result.config is config
    assert result.run_repository.root_path == str(tmp_path / "artifacts" / "pipeline-runs")
    assert result.splitter.is_drift_run is False
    assert result.trainer.is_drift_run is False


def test_factory_propagates_drift_mode(
    tmp_path,
) -> None:
    result = build_project_training_pipeline(
        {
            "paths": {
                "artifacts": str(tmp_path / "artifacts"),
            },
            "training": {
                "is_drift_run": True,
            },
        }
    )

    assert result.splitter.is_drift_run is True
    assert result.trainer.is_drift_run is True
