from unittest.mock import MagicMock, patch

import pytest

from mlops_sales_forecasting.tracking.mlflow import (
    log_evaluation_result,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
)


@patch(
    "mlops_sales_forecasting.tracking.mlflow.mlflow"
)
def test_logs_approved_evaluation(
    mlflow: MagicMock,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "run-1"
    mlflow.active_run.return_value = active_run

    evaluation = EvaluationResult(
        metrics={
            "accuracy": 0.91,
            "loss": 0.22,
        },
        approved=True,
    )

    result = log_evaluation_result(
        evaluation
    )

    assert result is evaluation
    mlflow.log_metrics.assert_called_once_with(
        {
            "accuracy": 0.91,
            "loss": 0.22,
        }
    )
    mlflow.set_tags.assert_called_once_with(
        {
            "evaluation.approved": "true",
        }
    )


@patch(
    "mlops_sales_forecasting.tracking.mlflow.mlflow"
)
def test_logs_rejected_evaluation(
    mlflow: MagicMock,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "run-2"
    mlflow.active_run.return_value = active_run

    evaluation = EvaluationResult(
        metrics={"accuracy": 0.60},
        approved=False,
        reasons=(
            "Accuracy below threshold.",
            "Calibration check failed.",
        ),
    )

    log_evaluation_result(evaluation)

    mlflow.log_metrics.assert_called_once_with(
        {"accuracy": 0.60}
    )
    mlflow.set_tags.assert_called_once_with(
        {
            "evaluation.approved": "false",
            "evaluation.reasons": (
                "Accuracy below threshold.; "
                "Calibration check failed."
            ),
        }
    )


@patch(
    "mlops_sales_forecasting.tracking.mlflow.mlflow"
)
def test_requires_active_mlflow_run(
    mlflow: MagicMock,
) -> None:
    mlflow.active_run.return_value = None

    evaluation = EvaluationResult(
        metrics={"accuracy": 0.91},
        approved=True,
    )

    with pytest.raises(
        RuntimeError,
        match="requires an active MLflow run",
    ):
        log_evaluation_result(evaluation)

    mlflow.log_metrics.assert_not_called()
    mlflow.set_tags.assert_not_called()


def test_requires_evaluation_result() -> None:
    with pytest.raises(
        TypeError,
        match="requires EvaluationResult",
    ):
        log_evaluation_result(
            MagicMock()
        )


@patch(
    "mlops_sales_forecasting.tracking.mlflow.mlflow"
)
def test_empty_metrics_still_logs_decision(
    mlflow: MagicMock,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "run-3"
    mlflow.active_run.return_value = active_run

    evaluation = EvaluationResult(
        metrics={},
        approved=True,
    )

    log_evaluation_result(evaluation)

    mlflow.log_metrics.assert_not_called()
    mlflow.set_tags.assert_called_once_with(
        {
            "evaluation.approved": "true",
        }
    )