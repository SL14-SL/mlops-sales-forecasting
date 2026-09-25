from unittest.mock import MagicMock

from mlops_sales_forecasting.inference.releases.contracts import (
    TaskType,
)
from mlops_sales_forecasting.inference.releases.project_input_provider import (
    RossmannServingReleaseInputProvider,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


def test_provider_builds_forecasting_release_input() -> None:
    training_result = TrainingResult(
        model=MagicMock(),
        run_id="run-123",
        metrics={
            "validation_rmse": 42.0,
        },
    )
    evaluation_result = EvaluationResult(
        metrics={
            "rmse": 42.0,
            "promo_rmse": 40.0,
            "non_promo_rmse": 44.0,
            "overall_bias": 2.0,
        },
        approved=True,
    )
    config = {
        "paths": {
            "validated_data": ("data/validation"),
            "features": "data/features",
            "models": "artifacts/models",
        },
        "model": {
            "type": "xgboost",
        },
        "training": {
            "target_transformation": "log1p",
        },
    }

    result = RossmannServingReleaseInputProvider().build_release_input(
        training_result=training_result,
        evaluation_result=(evaluation_result),
        config=config,
    )

    assert result.task_type is TaskType.FORECASTING
    assert result.model_type == "xgboost"
    assert set(result.sources) == {
        "store_metadata",
        "store_state",
        "known_calendar",
    }
    assert result.sources["store_metadata"].source_uri == "data/validation/store.parquet"
    assert result.sources["store_state"].source_uri == "artifacts/models/latest_state.json"
    assert result.sources["known_calendar"].source_uri == "data/features/known_calendar.parquet"
    assert result.metadata["target_transformation"] == "log1p"
