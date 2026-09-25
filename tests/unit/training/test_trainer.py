from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mlops_sales_forecasting.data.contracts import (
    DatasetSplits,
)
from mlops_sales_forecasting.training import trainer
from mlops_sales_forecasting.training.trainer import (
    RossmannModelTrainer,
)


def test_trainer_returns_template_training_result(
    monkeypatch,
) -> None:
    model = MagicMock()
    model.predict.return_value = np.log1p(
        [
            120.0,
            140.0,
        ]
    )

    build_model = MagicMock(return_value=model)
    fit_model = MagicMock()

    monkeypatch.setattr(
        trainer,
        "build_model",
        build_model,
    )
    monkeypatch.setattr(
        trainer,
        "fit_model",
        fit_model,
    )
    monkeypatch.setattr(
        trainer,
        "get_active_training_run_id",
        lambda: "run-123",
    )

    datasets = DatasetSplits(
        train=pd.DataFrame(
            {
                "Store": [
                    1,
                    1,
                ],
                "Date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                    ]
                ),
                "Sales": [
                    100.0,
                    110.0,
                ],
                "Customers": [
                    10,
                    11,
                ],
                "Promo": [
                    0,
                    1,
                ],
                "feature": [
                    1.0,
                    2.0,
                ],
            }
        ),
        validation=pd.DataFrame(
            {
                "Store": [
                    1,
                    1,
                ],
                "Date": pd.to_datetime(
                    [
                        "2026-01-03",
                        "2026-01-04",
                    ]
                ),
                "Sales": [
                    120.0,
                    140.0,
                ],
                "Customers": [
                    12,
                    14,
                ],
                "Promo": [
                    0,
                    1,
                ],
                "feature": [
                    3.0,
                    4.0,
                ],
            }
        ),
    )
    config = {
        "random_seed": 42,
        "data": {
            "target_column": "Sales",
            "known_targets": [
                "Sales",
                "Customers",
            ],
            "time_column": "Date",
        },
        "features": {
            "drop_columns": [],
        },
        "training": {
            "target_transformation": "log1p",
        },
        "metrics": {
            "evaluate_on_original_scale": True,
        },
        "model": {
            "type": "xgboost",
            "params": {
                "n_estimators": 10,
            },
        },
    }

    result = RossmannModelTrainer().train(
        datasets,
        config,
    )

    assert result.model is model
    assert result.run_id == "run-123"
    assert result.metrics["validation_rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result.parameters["model_type"] == "xgboost"
    assert result.parameters["target_transformation"] == "log1p"

    build_model.assert_called_once_with(
        config["model"],
        seed=42,
    )
    fit_model.assert_called_once()
