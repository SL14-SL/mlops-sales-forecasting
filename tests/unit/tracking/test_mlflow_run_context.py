from unittest.mock import MagicMock, patch

import pytest

from mlops_sales_forecasting.tracking.mlflow import (
    get_active_training_run_id,
)


@patch(
    "mlops_sales_forecasting.tracking.mlflow.mlflow"
)
def test_returns_active_training_run_id(
    mlflow: MagicMock,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "mlflow-run-123"
    mlflow.active_run.return_value = active_run

    run_id = get_active_training_run_id()

    assert run_id == "mlflow-run-123"
    mlflow.active_run.assert_called_once_with()


@patch(
    "mlops_sales_forecasting.tracking.mlflow.mlflow"
)
def test_requires_active_training_run(
    mlflow: MagicMock,
) -> None:
    mlflow.active_run.return_value = None

    with pytest.raises(
        RuntimeError,
        match="requires an active MLflow run",
    ):
        get_active_training_run_id()