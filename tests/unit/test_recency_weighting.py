import numpy as np
import pandas as pd

from src.training.train import build_recency_weights


def test_build_recency_weights_only_weights_promo_rows():
    latest_date = pd.Timestamp("2015-06-23")

    dates = pd.Series(
        [
            latest_date,
            latest_date - pd.Timedelta(days=30),
            latest_date - pd.Timedelta(days=31),
            latest_date - pd.Timedelta(days=60),
            latest_date - pd.Timedelta(days=61),
            latest_date - pd.Timedelta(days=120),
            latest_date - pd.Timedelta(days=121),
        ]
    )

    promo_values = pd.Series(
        [
            1,
            0,
            1,
            0,
            1,
            1,
            1,
        ]
    )

    config = {
        "last_30_days_weight": 10.0,
        "last_60_days_weight": 5.0,
        "last_120_days_weight": 2.0,
        "default_weight": 1.0,
    }

    weights = build_recency_weights(
        dates=dates,
        promo_values=promo_values,
        weighting_config=config,
    )

    expected = np.array(
        [
            10.0,
            1.0,
            5.0,
            1.0,
            2.0,
            2.0,
            1.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        weights,
        expected,
    )