from unittest.mock import MagicMock

from mlops_sales_forecasting.training import model_logger
from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)
from mlops_sales_forecasting.training.model_logger import (
    XGBoostModelArtifactLogger,
)


def test_xgboost_logger_returns_run_model_uri(
    monkeypatch,
) -> None:
    log_model = MagicMock()

    monkeypatch.setattr(
        model_logger.mlflow.xgboost,
        "log_model",
        log_model,
    )

    trained_model = MagicMock()
    result = TrainingResult(
        model=trained_model,
        run_id="run-123",
        metrics={
            "validation_rmse": 12.5,
        },
    )
    config = {
        "model": {
            "type": "xgboost",
        },
        "data": {
            "target_column": "Sales",
        },
        "training": {
            "target_transformation": "log1p",
        },
    }

    model_uri = XGBoostModelArtifactLogger().log_model(
        result,
        artifact_path="model",
        config=config,
    )

    assert model_uri == ("runs:/run-123/model")

    log_model.assert_called_once_with(
        trained_model,
        name="model",
        metadata={
            "model_type": "xgboost",
            "target_column": "Sales",
            "target_transformation": "log1p",
        },
    )


def test_xgboost_logger_rejects_other_model_type() -> None:
    result = TrainingResult(
        model=MagicMock(),
        run_id="run-123",
        metrics={},
    )

    try:
        XGBoostModelArtifactLogger().log_model(
            result,
            artifact_path="model",
            config={
                "model": {
                    "type": "random_forest",
                },
            },
        )
    except ValueError as error:
        assert "model.type='xgboost'" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")
