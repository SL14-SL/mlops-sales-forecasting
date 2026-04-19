import pandas as pd

from src.data.features.common import (
    add_basic_calendar_features,
    cast_object_columns_to_category,
    drop_columns_if_present,
    ensure_datetime_column,
)


def test_ensure_datetime_column_parses_date():
    df = pd.DataFrame({"Date": ["2026-03-06"]})

    result = ensure_datetime_column(df, "Date")

    assert str(result["Date"].dtype).startswith("datetime64")


def test_add_basic_calendar_features():
    df = pd.DataFrame({"Date": ["2026-03-06"]})

    result = add_basic_calendar_features(df, date_column="Date")

    assert "DayOfWeek" in result.columns
    assert "WeekOfYear" in result.columns
    assert "day" in result.columns
    assert "month" in result.columns
    assert "year" in result.columns
    assert "is_month_start" in result.columns
    assert "is_month_end" in result.columns

    assert result.loc[0, "day"] == 6
    assert result.loc[0, "month"] == 3
    assert result.loc[0, "year"] == 2026


def test_cast_object_columns_to_category():
    df = pd.DataFrame(
        {
            "StateHoliday": ["0"],
            "StoreType": ["a"],
            "Customers": [100],
        }
    )

    result = cast_object_columns_to_category(df)

    assert str(result["StateHoliday"].dtype) == "category"
    assert str(result["StoreType"].dtype) == "category"
    assert result["Customers"].dtype == df["Customers"].dtype


def test_drop_columns_if_present():
    df = pd.DataFrame(
        {
            "A": [1],
            "B": [2],
            "C": [3],
        }
    )

    result = drop_columns_if_present(df, ["B", "X"])

    assert "B" not in result.columns
    assert "A" in result.columns
    assert "C" in result.columns