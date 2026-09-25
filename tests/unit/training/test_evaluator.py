from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mlops_sales_forecasting.data.contracts import (
    DatasetSplits,
)
from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)
from mlops_sales_forecasting.training.evaluator import (
    RossmannModelEvaluator,
)


def datasets() -> DatasetSplits:
    train = pd.DataFrame(
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
                90.0,
                100.0,
            ],
            "Customers": [
                9,
                10,
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
    )
    validation = pd.DataFrame(
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
    )

    return DatasetSplits(
        train=train,
        validation=validation,
    )


def config(
    *,
    minimum_validation_rows: int = 2,
) -> dict:
    return {
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
        "promotion": {
            "minimum_validation_rows": (minimum_validation_rows),
            "minimum_segment_rows": 1,
            "required_segments": [
                "promo",
                "non_promo",
            ],
        },
    }


def training_result() -> TrainingResult:
    model = MagicMock()
    model.get_booster.return_value.feature_names = []
    model.predict.return_value = np.log1p(
        [
            120.0,
            140.0,
        ]
    )

    return TrainingResult(
        model=model,
        run_id="run-123",
        metrics={
            "validation_rmse": 0.0,
        },
    )


def test_evaluator_approves_valid_candidate() -> None:
    result = RossmannModelEvaluator().evaluate(
        training_result(),
        datasets(),
        config(),
    )

    assert result.approved is True
    assert result.reasons == ()
    assert result.metrics["rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result.metrics["promo_rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result.metrics["non_promo_rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_evaluator_rejects_insufficient_validation_rows() -> None:
    result = RossmannModelEvaluator().evaluate(
        training_result(),
        datasets(),
        config(minimum_validation_rows=3),
    )

    assert result.approved is False
    assert result.reasons == ("Validation dataset contains fewer rows than required.",)
