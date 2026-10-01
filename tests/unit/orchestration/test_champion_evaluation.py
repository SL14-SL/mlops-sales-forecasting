from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.data.contracts import (
    DatasetSplits,
)
from mlops_sales_forecasting.inference.model_loader import (
    LoadedXGBoostModel,
)
from mlops_sales_forecasting.orchestration.champion_evaluation import (
    build_champion_metrics_provider,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
)


def test_provider_evaluates_champion_on_current_splits() -> None:
    estimator = MagicMock()
    loaded_model = LoadedXGBoostModel(
        estimator=estimator,
        input_columns=("feature",),
    )
    model_loader = MagicMock(return_value=loaded_model)
    evaluator = MagicMock()
    datasets = MagicMock(spec=DatasetSplits)
    config = {
        "data": {
            "target_column": "Sales",
        },
    }

    evaluator.evaluate.return_value = EvaluationResult(
        metrics={
            "rmse": 1200.0,
            "promo_rmse": 1400.0,
            "non_promo_rmse": 900.0,
        },
        approved=True,
    )

    provider = build_champion_metrics_provider(
        evaluator=evaluator,
        datasets=datasets,
        config=config,
        model_loader=model_loader,
    )

    result = provider("models:/forecast-model/5")

    assert result == {
        "rmse": 1200.0,
        "promo_rmse": 1400.0,
        "non_promo_rmse": 900.0,
    }
    model_loader.assert_called_once_with("models:/forecast-model/5")

    training_result = evaluator.evaluate.call_args.args[0]

    assert training_result.model is estimator
    assert training_result.run_id == ("paired-champion-evaluation")
    assert training_result.metrics == {}

    evaluator.evaluate.assert_called_once_with(
        training_result,
        datasets,
        config,
    )


def test_provider_rejects_failed_champion_evaluation() -> None:
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        metrics={
            "rmse": 1200.0,
        },
        approved=False,
        reasons=("Validation dataset contains fewer rows than required.",),
    )
    model_loader = MagicMock(
        return_value=LoadedXGBoostModel(
            estimator=MagicMock(),
            input_columns=("feature",),
        )
    )

    provider = build_champion_metrics_provider(
        evaluator=evaluator,
        datasets=MagicMock(spec=DatasetSplits),
        config={},
        model_loader=model_loader,
    )

    with pytest.raises(
        ValueError,
        match=("Champion could not be evaluated"),
    ):
        provider("models:/forecast-model/5")


@pytest.mark.parametrize(
    "model_uri",
    [
        "",
        "   ",
    ],
)
def test_provider_rejects_empty_model_uri(
    model_uri: str,
) -> None:
    provider = build_champion_metrics_provider(
        evaluator=MagicMock(),
        datasets=MagicMock(spec=DatasetSplits),
        config={},
        model_loader=MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="Champion model URI",
    ):
        provider(model_uri)
