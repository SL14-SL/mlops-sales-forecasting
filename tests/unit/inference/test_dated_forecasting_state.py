import pandas as pd
import pytest

from mlops_sales_forecasting.data.features.dated_state import (
    build_dated_feature_state,
)
from mlops_sales_forecasting.inference.forecasting_policy import (
    inject_forecasting_state_features,
)


@pytest.fixture
def config() -> dict:
    return {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": ["Store"],
        },
        "features": {
            "lag_features": {
                "lags": [1, 3],
                "rolling_windows": [3],
            },
        },
    }


@pytest.fixture
def state() -> dict:
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


def test_serving_injects_dated_features_without_reordering(
    config: dict,
    state: dict,
) -> None:
    frame = pd.DataFrame(
        {
            "Store": [2, 1],
            "Date": pd.to_datetime(["2026-01-04", "2026-01-04"]),
            "Open": [1, 0],
            "sales_lag_1": [0.0, 0.0],
        },
        index=[9, 9],
    )

    result = inject_forecasting_state_features(frame, state, config)

    assert result["Store"].tolist() == [2, 1]
    assert result["Open"].tolist() == [1, 0]
    assert result["sales_lag_1"].tolist() == [300.0, 30.0]
    assert result["sales_lag_3"].tolist() == [100.0, 10.0]
    assert result["sales_rolling_mean_3"].tolist() == [200.0, 20.0]
    assert result.index.tolist() == [9, 9]
    assert result["Date"].equals(frame["Date"])
    assert frame["sales_lag_1"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("forecast_date", ["2026-01-03", "2026-01-05"])
def test_serving_rejects_unsupported_forecast_dates(
    config: dict,
    state: dict,
    forecast_date: str,
) -> None:
    frame = pd.DataFrame(
        {"Store": [1], "Date": [forecast_date]}
    )

    with pytest.raises(ValueError, match="only next-day forecasts"):
        inject_forecasting_state_features(frame, state, config)


def test_serving_rejects_missing_calendar_history(
    config: dict,
    state: dict,
) -> None:
    state["stores"]["1"].pop(1)
    frame = pd.DataFrame(
        {"Store": [1], "Date": ["2026-01-04"]}
    )

    with pytest.raises(ValueError, match="Missing dated history"):
        inject_forecasting_state_features(frame, state, config)


@pytest.mark.parametrize("schema_version", [2, True])
def test_serving_rejects_unsupported_state_versions(
    config: dict,
    state: dict,
    schema_version: int | bool,
) -> None:
    state["schema_version"] = schema_version
    frame = pd.DataFrame(
        {"Store": [1], "Date": ["2026-01-04"]}
    )

    with pytest.raises(ValueError, match="schema_version=1"):
        inject_forecasting_state_features(frame, state, config)