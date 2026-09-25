from unittest.mock import MagicMock, patch

import pytest

from mlops_sales_forecasting.tracking.registry import (
    register_approved_model,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)

CONFIG = {
    "tracking": {
        "model_name": "example-model",
    }
}


def build_training_result() -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id="run-7",
        metrics={"score": 0.91},
    )


def build_evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        metrics={"score": 0.91},
        approved=True,
    )


@patch(
    "mlops_sales_forecasting.tracking.registry."
    "mlflow"
)
def test_registers_logged_model_uri(
    mlflow: MagicMock,
) -> None:
    model_version = MagicMock()
    model_version.version = "7"
    mlflow.register_model.return_value = (
        model_version
    )

    result = register_approved_model(
        training_result=(
            build_training_result()
        ),
        evaluation_result=(
            build_evaluation_result()
        ),
        config=CONFIG,
        model_uri="models:/m-logged-model",
    )

    mlflow.register_model.assert_called_once_with(
        model_uri="models:/m-logged-model",
        name="example-model",
    )

    assert result.registered is True
    assert result.model_version == "7"
    assert (
        result.model_uri
        == "models:/example-model/7"
    )


@patch(
    "mlops_sales_forecasting.tracking.registry."
    "mlflow"
)
def test_rejects_empty_logged_model_uri(
    mlflow: MagicMock,
) -> None:
    with pytest.raises(
        ValueError,
        match="MLflow model URI",
    ):
        register_approved_model(
            training_result=(
                build_training_result()
            ),
            evaluation_result=(
                build_evaluation_result()
            ),
            config=CONFIG,
            model_uri="",
        )

    mlflow.register_model.assert_not_called()


@patch(
    "mlops_sales_forecasting.tracking.registry."
    "mlflow"
)
def test_rejected_candidate_ignores_model_uri(
    mlflow: MagicMock,
) -> None:
    evaluation = EvaluationResult(
        metrics={"score": 0.40},
        approved=False,
        reasons=("Score below threshold.",),
    )

    result = register_approved_model(
        training_result=(
            build_training_result()
        ),
        evaluation_result=evaluation,
        config=CONFIG,
        model_uri="",
    )

    assert result.registered is False
    mlflow.register_model.assert_not_called()