from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.training import model_logger
from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)
from mlops_sales_forecasting.training.model_logger import (
    XGBoostModelArtifactLogger,
)


def test_xgboost_logger_returns_logged_model_uri(
    monkeypatch,
) -> None:
    log_model = MagicMock()
    log_model.return_value.model_uri = (
        "models:/m-test-model"
    )
    monkeypatch.setattr(
        model_logger.mlflow.xgboost,
        "log_model",
        log_model,
    )

    trained_model = MagicMock()
    trained_model.predict.return_value = [
        100.0,
    ]
    input_example = pd.DataFrame(
        [
            {
                "Store": 1,
                "Promo": 0,
                "feature": 2.5,
            },
        ]
    )
    signature = MagicMock()
    infer_signature = MagicMock(
        return_value=signature
    )

    monkeypatch.setattr(
        model_logger,
        "infer_signature",
        infer_signature,
    )
    result = TrainingResult(
        model=trained_model,
        run_id="run-123",
        metrics={
            "validation_rmse": 12.5,
        },
        input_example=input_example,
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

    assert model_uri == "models:/m-test-model"

    infer_signature.assert_called_once()

    signature_call = infer_signature.call_args
    pd.testing.assert_frame_equal(
        signature_call.args[0],
        input_example,
    )
    assert signature_call.args[1] == [
        100.0,
    ]

    log_model.assert_called_once_with(
        trained_model,
        name="model",
        signature=signature,
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

def test_xgboost_logger_requires_input_example() -> None:
    result = TrainingResult(
        model=MagicMock(),
        run_id="run-123",
        metrics={},
    )

    with pytest.raises(
        ValueError,
        match="non-empty pandas input example",
    ):
        XGBoostModelArtifactLogger().log_model(
            result,
            artifact_path="model",
            config={
                "model": {
                    "type": "xgboost",
                },
            },
        )