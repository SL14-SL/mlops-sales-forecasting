import pandas as pd
import pytest

from mlops_sales_forecasting.data.features.dated_state import (
    build_dated_feature_state,
    build_dated_lag_features,
)


def example_state() -> dict:
    return build_dated_feature_state(
        pd.DataFrame(
            {
                "Store": [1, 1, 1, 2, 2, 2],
                "Date": [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ],
                "Sales": [10, 20, 30, 100, 200, 300],
            }
        ),
        history_length=3,
    )


def test_lags_use_calendar_dates_and_preserve_batch_order() -> None:
    inputs = pd.DataFrame(
        {
            "Store": [2, 1],
            "Date": ["2026-01-04", "2026-01-04"],
        },
        index=[9, 9],
    )

    result = build_dated_lag_features(
        inputs,
        example_state(),
        lags=[1, 3],
        rolling_windows=[3],
    )

    assert result["sales_lag_1"].tolist() == [300.0, 30.0]
    assert result["sales_lag_3"].tolist() == [100.0, 10.0]
    assert result["sales_rolling_mean_3"].tolist() == [200.0, 20.0]
    assert result.index.tolist() == [9, 9]


@pytest.mark.parametrize(
    "forecast_date",
    [
        "2026-01-03",
        "2026-01-05",
        "2026-02-01",
    ],
)
def test_rejects_dates_outside_next_day(forecast_date: str) -> None:
    inputs = pd.DataFrame(
        {"Store": [1], "Date": [forecast_date]}
    )

    with pytest.raises(ValueError, match="only next-day forecasts"):
        build_dated_lag_features(
            inputs,
            example_state(),
            lags=[1],
            rolling_windows=[3],
        )


def test_missing_calendar_day_is_not_replaced_by_another_observation() -> None:
    state = {
        "schema_version": 1,
        "stores": {
            "1": [
                {"date": "2026-01-01", "value": 10.0},
                {"date": "2026-01-03", "value": 30.0},
            ],
        },
    }
    inputs = pd.DataFrame(
        {"Store": [1], "Date": ["2026-01-04"]}
    )

    with pytest.raises(ValueError, match="Missing dated history"):
        build_dated_lag_features(
            inputs,
            state,
            lags=[1, 2],
            rolling_windows=[3],
        )


def test_rejects_store_without_history() -> None:
    inputs = pd.DataFrame(
        {"Store": [3], "Date": ["2026-01-04"]}
    )

    with pytest.raises(ValueError, match="No dated history"):
        build_dated_lag_features(
            inputs,
            example_state(),
            lags=[1],
            rolling_windows=[3],
        )