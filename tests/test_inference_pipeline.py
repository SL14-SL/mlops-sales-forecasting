import pandas as pd
import pytest

from src.inference.pipeline import (
    validate_prediction_input,
    align_features_for_model,
    apply_business_rules,
)


def test_validate_prediction_input(sample_prediction_df):
    result = validate_prediction_input(sample_prediction_df)

    assert "Store" in result.columns
    assert result["Store"].dtype == int

def test_validate_prediction_input_rejects_target_column():
    df = pd.DataFrame([{
        "Store": 1,
        "Date": "2026-03-24",
        "Open": 1,
        "Promo": 0,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
        "Sales": 1234,
    }])

    with pytest.raises(ValueError, match="must not contain target column"):
        validate_prediction_input(df)

def test_validate_prediction_input_requires_store():
    df = pd.DataFrame([{
        "Date": "2026-03-24",
        "Open": 1,
        "Promo": 0,
    }])

    with pytest.raises(Exception):
        validate_prediction_input(df)


def test_validate_prediction_input_rejects_invalid_open_flag():
    df = pd.DataFrame([{
        "Store": 1,
        "Date": "2026-03-24",
        "Open": 2,
        "Promo": 0,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
    }])

    with pytest.raises(Exception):
        validate_prediction_input(df)


def test_align_features_for_xgboost(mock_xgb_model):
    import pandas as pd

    processed_df = pd.DataFrame(
        [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Customers": 500,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
                "StoreType": "a",
                "Assortment": "a",
                "CompetitionDistance": 500.0,
                "Promo2": 1,
                "WeekOfYear": 10,
                "day": 6,
                "month": 3,
                "year": 2026,
                "is_month_start": 0,
                "is_month_end": 0,
                "sales_lag_1": 1600.0,
                "sales_lag_7": 1000.0,
                "sales_rolling_mean_7": 1300.0,
                "extra_column": 999,
            }
        ]
    )

    result = align_features_for_model(processed_df, mock_xgb_model, "xgboost")

    assert list(result.columns) == mock_xgb_model.get_booster.return_value.feature_names
    assert "extra_column" not in result.columns


def test_apply_business_rules_closed_store():
    result = apply_business_rules(1234.5, is_open=0)
    assert result == 0.0


def test_apply_business_rules_open_store():
    result = apply_business_rules(1234.5, is_open=1)
    assert result == 1234.5