from types import SimpleNamespace
from unittest.mock import MagicMock
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
    INTERNAL_ERROR
)
import numpy as np
import pandas as pd
import pytest

import src.training.evaluate as evaluate

from src.training.evaluate import (
    align_features_for_evaluation,
)

from mlflow.exceptions import MlflowException

class FakeBooster:
    def __init__(self, feature_names):
        self.feature_names = feature_names


class FakeModel:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def get_booster(self):
        return FakeBooster(
            self.feature_names
        )


def test_align_features_for_evaluation_removes_extra_columns():
    model = FakeModel(
        [
            "Store",
            "Promo",
        ]
    )

    features = pd.DataFrame(
        {
            "Store": [1],
            "calendar_feature": [3],
            "Promo": [1],
        }
    )

    result = align_features_for_evaluation(
        model,
        features,
    )

    assert result.columns.tolist() == [
        "Store",
        "Promo",
    ]


def test_align_features_for_evaluation_rejects_missing_columns():
    model = FakeModel(
        [
            "Store",
            "Promo",
        ]
    )

    features = pd.DataFrame(
        {
            "Store": [1],
        }
    )

    try:
        align_features_for_evaluation(
            model,
            features,
        )
    except ValueError as error:
        assert "Promo" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )

def _run_metadata(target_transformation: str = "none"):
    return SimpleNamespace(
        data=SimpleNamespace(
            tags={
                "target_transformation": target_transformation,
            },
            params={},
        ),
    )


def _model(predictions: list[float]):
    model = MagicMock()
    model.get_booster.return_value.feature_names = []
    model.predict.return_value = np.asarray(predictions)
    return model


def test_compare_models_blocks_promotion_when_champion_cannot_be_loaded(
    monkeypatch,
):
    validation_data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "Sales": [100.0, 200.0, 300.0],
        }
    )

    challenger = _model(
        [100.0, 200.0, 300.0],
    )

    monkeypatch.setattr(
        evaluate.pd,
        "read_parquet",
        MagicMock(return_value=validation_data),
    )

    load_model = MagicMock(
        side_effect=[
            challenger,
            RuntimeError("MLflow registry unavailable"),
        ]
    )

    monkeypatch.setattr(
        evaluate.mlflow.xgboost,
        "load_model",
        load_model,
    )

    client = MagicMock()
    client.get_run.return_value = _run_metadata()

    monkeypatch.setattr(
        evaluate,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    monkeypatch.setattr(
        evaluate,
        "build_drop_columns",
        MagicMock(return_value=[]),
    )

    monkeypatch.setitem(
        evaluate.TRAIN_CFG,
        "data",
        {
            **evaluate.TRAIN_CFG["data"],
            "target_column": "Sales",
        },
    )

    with pytest.raises(
        evaluate.ModelComparisonError,
        match="promotion was blocked",
    ):
        evaluate.compare_models(
            new_run_id="candidate-run-123",
            val_path="validation.parquet",
        )

    assert load_model.call_count == 2

def test_champion_exists_returns_true_when_alias_is_present(
    monkeypatch,
):
    client = MagicMock()
    client.get_registered_model.return_value = (
        SimpleNamespace(
            aliases={
                "champion": "3",
            }
        )
    )

    monkeypatch.setattr(
        evaluate,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    assert evaluate.champion_exists() is True

    client.get_registered_model.assert_called_once_with(
        evaluate.MODEL_NAME
    )

def test_champion_exists_returns_false_when_alias_is_missing(
    monkeypatch,
):
    client = MagicMock()
    client.get_registered_model.return_value = (
        SimpleNamespace(
            aliases={
                "challenger": "2",
            }
        )
    )

    monkeypatch.setattr(
        evaluate,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    assert evaluate.champion_exists() is False

def test_champion_exists_returns_false_when_model_is_missing(
    monkeypatch,
):
    client = MagicMock()
    client.get_registered_model.side_effect = (
        MlflowException(
            "Registered model not found.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    )

    monkeypatch.setattr(
        evaluate,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    assert evaluate.champion_exists() is False

def test_champion_exists_propagates_registry_errors(
    monkeypatch,
):
    client = MagicMock()
    client.get_registered_model.side_effect = (
        MlflowException(
            "Registry unavailable.",
            error_code=INTERNAL_ERROR,
        )
    )

    monkeypatch.setattr(
        evaluate,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    with pytest.raises(
        MlflowException,
        match="Registry unavailable",
    ):
        evaluate.champion_exists()

    