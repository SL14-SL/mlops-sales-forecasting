import numpy as np
import pandas as pd

from mlops_sales_forecasting.training.weighting import build_recency_weights


def test_build_recency_weights_only_weights_promo_rows():
    latest_date = pd.Timestamp("2015-06-23")

    dates = pd.Series(
        [
            latest_date,
            latest_date - pd.to_timedelta(30, unit="D"),
            latest_date - pd.to_timedelta(31, unit="D"),
            latest_date - pd.to_timedelta(60, unit="D"),
            latest_date - pd.to_timedelta(61, unit="D"),
            latest_date - pd.to_timedelta(120, unit="D"),
            latest_date - pd.to_timedelta(121, unit="D"),
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
