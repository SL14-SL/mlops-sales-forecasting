import pandas as pd
import pytest

from src.data.features.build_features import build_features, preprocess_data


@pytest.fixture
def training_config() -> dict:
    return {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": ["Store"],
        },
        "features": {
            "enabled_steps": [
                "sort",
                "temporal",
                "lags",
                "competition",
                "promo",
                "cast_categoricals",
                "drop_technical",
                "drop_configured",
            ],
            "technical_drop_columns": [
                "CompetitionOpenSinceYear",
                "CompetitionOpenSinceMonth",
                "Promo2SinceYear",
                "Promo2SinceWeek",
                "PromoInterval",
            ],
            "drop_columns": [],
        },
    }


def test_build_features_training_mode(training_config):
    df = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1, 1, 1, 1, 1],
            "Date": pd.date_range("2026-03-01", periods=8, freq="D"),
            "Sales": [10, 20, 30, 40, 50, 60, 70, 80],
            "StateHoliday": ["0"] * 8,
            "StoreType": ["a"] * 8,
            "Assortment": ["a"] * 8,
            "CompetitionOpenSinceYear": [2025] * 8,
            "CompetitionOpenSinceMonth": [1] * 8,
            "Promo2SinceYear": [2025] * 8,
            "Promo2SinceWeek": [2] * 8,
            "PromoInterval": ["Jan,Apr,Jul,Oct"] * 8,
        }
    )

    result = build_features(df, config=training_config, mode="train")

    assert "WeekOfYear" in result.columns
    assert "sales_lag_1" in result.columns
    assert "sales_lag_7" in result.columns
    assert "sales_rolling_mean_7" in result.columns
    assert "CompetitionOpen" in result.columns
    assert "PromoOpen" in result.columns

    assert "CompetitionOpenSinceYear" not in result.columns
    assert "CompetitionOpenSinceMonth" not in result.columns
    assert "Promo2SinceYear" not in result.columns
    assert "Promo2SinceWeek" not in result.columns
    assert "PromoInterval" not in result.columns

    assert str(result["StateHoliday"].dtype) == "category"
    assert str(result["StoreType"].dtype) == "category"
    assert str(result["Assortment"].dtype) == "category"


def test_build_features_auto_mode_with_sales_uses_training_logic(training_config):
    df = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.date_range("2026-03-01", periods=2, freq="D"),
            "Sales": [10, 20],
            "StateHoliday": ["0", "0"],
        }
    )

    result = build_features(df, config=training_config, mode="auto")

    assert "sales_lag_1" in result.columns
    assert result.loc[0, "sales_lag_1"] == 0
    assert result.loc[1, "sales_lag_1"] == 10


def test_build_features_auto_mode_without_sales_uses_inference_logic(training_config):
    df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-03-06"],
            "StateHoliday": ["0"],
        }
    )

    result = build_features(df, config=training_config, mode="auto")

    assert result.loc[0, "sales_lag_1"] == 0.0
    assert result.loc[0, "sales_lag_7"] == 0.0
    assert result.loc[0, "sales_rolling_mean_7"] == 0.0


def test_build_features_handles_empty_dataframe(training_config):
    df = pd.DataFrame(columns=["Store", "Date", "StateHoliday"])

    result = build_features(df, config=training_config, mode="auto")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_build_features_invalid_mode_raises(training_config):
    df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-03-06"],
        }
    )

    with pytest.raises(ValueError):
        build_features(df, config=training_config, mode="invalid")


def test_build_features_explicit_inference_mode_overrides_sales_column(training_config):
    df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-03-06"],
            "Sales": [999],
            "StateHoliday": ["0"],
        }
    )

    result = build_features(df, config=training_config, mode="inference")

    assert result.loc[0, "sales_lag_1"] == 0.0
    assert result.loc[0, "sales_lag_7"] == 0.0
    assert result.loc[0, "sales_rolling_mean_7"] == 0.0


def test_build_features_respects_enabled_steps(training_config):
    cfg = {
        "data": training_config["data"],
        "features": {
            "enabled_steps": ["cast_categoricals"],
            "technical_drop_columns": [],
            "drop_columns": [],
        },
    }

    df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-03-06"],
            "Sales": [10],
            "StateHoliday": ["0"],
            "StoreType": ["a"],
        }
    )

    result = build_features(df, config=cfg, mode="train")

    assert "sales_lag_1" not in result.columns
    assert "WeekOfYear" not in result.columns
    assert str(result["StateHoliday"].dtype) == "category"
    assert str(result["StoreType"].dtype) == "category"


def test_build_features_uses_configured_column_names():
    cfg = {
        "data": {
            "target_column": "y",
            "time_column": "ds",
            "id_columns": ["entity_id"],
        },
        "features": {
            "enabled_steps": [
                "sort",
                "temporal",
                "lags",
                "cast_categoricals",
            ],
            "technical_drop_columns": [],
            "drop_columns": [],
        },
    }

    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 1, 1, 1, 1, 1],
            "ds": pd.date_range("2026-03-01", periods=8, freq="D"),
            "y": [10, 20, 30, 40, 50, 60, 70, 80],
            "category_col": ["a"] * 8,
        }
    )

    result = build_features(df, config=cfg, mode="train")

    assert len(result) == 8
    assert "WeekOfYear" in result.columns
    assert "y_lag_1" in result.columns
    assert "y_lag_7" in result.columns
    assert "y_rolling_mean_7" in result.columns
    assert "sales_lag_1" not in result.columns
    assert str(result["category_col"].dtype) == "category"


def test_preprocess_data_wrapper_still_works():
    df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-03-06"],
            "StateHoliday": ["0"],
        }
    )

    result = preprocess_data(df, mode="inference")

    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "sales_lag_1"] == 0.0
    assert result.loc[0, "sales_lag_7"] == 0.0
    assert result.loc[0, "sales_rolling_mean_7"] == 0.0


def test_build_features_uses_configured_lag_parameters():
    cfg = {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": ["Store"],
        },
        "features": {
            "enabled_steps": ["sort", "lags"],
            "lag_features": {
                "lags": [1, 3],
                "rolling_windows": [2],
            },
            "technical_drop_columns": [],
            "drop_columns": [],
        },
    }

    df = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1],
            "Date": pd.date_range("2026-03-01", periods=4, freq="D"),
            "Sales": [10, 20, 30, 40],
        }
    )

    result = build_features(df, config=cfg, mode="train")

    assert "sales_lag_1" in result.columns
    assert "sales_lag_3" in result.columns
    assert "sales_rolling_mean_2" in result.columns

    assert "sales_lag_7" not in result.columns
    assert "sales_rolling_mean_7" not in result.columns
    

def test_build_features_inference_mode_uses_dynamic_lag_names():
    cfg = {
        "data": {
            "target_column": "y",
            "time_column": "ds",
            "id_columns": ["entity_id"],
        },
        "features": {
            "enabled_steps": ["lags"],
            "technical_drop_columns": [],
            "drop_columns": [],
        },
    }

    df = pd.DataFrame(
        {
            "entity_id": [1],
            "ds": ["2026-03-06"],
        }
    )

    result = build_features(df, config=cfg, mode="inference")

    assert result.loc[0, "y_lag_1"] == 0.0
    assert result.loc[0, "y_lag_7"] == 0.0
    assert result.loc[0, "y_rolling_mean_7"] == 0.0