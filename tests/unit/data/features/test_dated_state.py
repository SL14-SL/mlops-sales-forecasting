import pandas as pd
import pytest

from mlops_sales_forecasting.data.features.dated_state import (
    build_dated_feature_state,
)


def test_state_preserves_dates_and_limits_history_per_store() -> None:
    frame = pd.DataFrame(
        {
            "Store": [2, 1, 1, 1],
            "Date": [
                "2026-01-02",
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
            ],
            "Sales": [200, 140, 100, 120],
        }
    )

    result = build_dated_feature_state(frame, history_length=2)

    assert result == {
        "schema_version": 1,
        "stores": {
            "1": [
                {"date": "2026-01-02", "value": 120.0},
                {"date": "2026-01-03", "value": 140.0},
            ],
            "2": [
                {"date": "2026-01-02", "value": 200.0},
            ],
        },
    }

    # Building state must not modify the source dataframe.
    assert frame["Date"].iloc[0] == "2026-01-02"


def test_state_keeps_calendar_gaps_visible() -> None:
    frame = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2026-01-01", "2026-01-03"],
            "Sales": [100, 140],
        }
    )

    result = build_dated_feature_state(frame, history_length=7)

    assert result["stores"]["1"] == [
        {"date": "2026-01-01", "value": 100.0},
        {"date": "2026-01-03", "value": 140.0},
    ]


def test_state_rejects_duplicate_store_dates() -> None:
    frame = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2026-01-01", "2026-01-01"],
            "Sales": [100, 120],
        }
    )

    with pytest.raises(ValueError, match="Duplicate store/date"):
        build_dated_feature_state(frame, history_length=7)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_state_rejects_invalid_sales(value: float) -> None:
    frame = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-01-01"],
            "Sales": [value],
        }
    )

    with pytest.raises(ValueError):
        build_dated_feature_state(frame, history_length=7)


@pytest.mark.parametrize("history_length", [0, -1, True, 1.5])
def test_state_rejects_invalid_history_length(history_length) -> None:
    frame = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-01-01"],
            "Sales": [100],
        }
    )

    with pytest.raises(ValueError, match="positive integer"):
        build_dated_feature_state(frame, history_length=history_length)