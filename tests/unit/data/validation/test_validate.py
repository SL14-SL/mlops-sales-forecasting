import pandas as pd
import pandera.pandas as pa
import pytest

from mlops_sales_forecasting.data.validation.validate import (
    validate_store,
    validate_train,
)


@pytest.fixture
def valid_sales_df():
    return pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-02-26"],
            "Sales": [100.0],
            "Customers": [10],
            "Open": [1],
            "Promo": [1],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
            "DayOfWeek": [4],
        }
    )


@pytest.fixture
def valid_store_df():
    return pd.DataFrame(
        {
            "Store": [1],
            "StoreType": ["a"],
            "Assortment": ["a"],
            "CompetitionDistance": [500.0],
        }
    )


def test_validate_train_happy_path(valid_sales_df):
    result_df = validate_train(valid_sales_df)

    assert not result_df.empty
    assert "Store" in result_df.columns


@pytest.mark.parametrize(
    "invalid_col, bad_value",
    [
        ("Sales", -1.0),
        ("Open", 5),
        ("StateHoliday", "Z"),
        ("Store", 0),
    ],
)
def test_validate_train_logic_errors(valid_sales_df, invalid_col, bad_value):
    df = valid_sales_df.copy()
    df[invalid_col] = [bad_value]

    with pytest.raises(pa.errors.SchemaError):
        validate_train(df)


def test_validate_train_missing_column(valid_sales_df):
    df = valid_sales_df.drop(columns=["Customers"])

    with pytest.raises(pa.errors.SchemaError):
        validate_train(df)


def test_validate_train_coerces_date_column(valid_sales_df):
    result_df = validate_train(valid_sales_df)

    assert str(result_df["Date"].dtype).startswith("datetime64")


def test_validate_train_allows_optional_columns(
    valid_sales_df,
):
    dataframe = valid_sales_df.drop(
        columns=[
            "Sales",
            "DayOfWeek",
        ]
    )

    result_df = validate_train(
        dataframe
    )

    assert "Sales" not in result_df.columns
    assert "DayOfWeek" not in result_df.columns
    

def test_validate_store_happy_path(valid_store_df):
    result_df = validate_store(valid_store_df)

    assert not result_df.empty
    assert "StoreType" in result_df.columns


@pytest.mark.parametrize(
    "invalid_col, bad_value",
    [
        ("Store", 0),
        ("StoreType", "z"),
        ("Assortment", "z"),
        ("CompetitionDistance", -1.0),
    ],
)
def test_validate_store_logic_errors(valid_store_df, invalid_col, bad_value):
    df = valid_store_df.copy()
    df[invalid_col] = [bad_value]

    with pytest.raises(pa.errors.SchemaError):
        validate_store(df)


def test_validate_store_missing_column(valid_store_df):
    df = valid_store_df.drop(columns=["StoreType"])

    with pytest.raises(pa.errors.SchemaError):
        validate_store(df)