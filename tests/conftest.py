import os

os.environ["PREFECT_API_MODE"] = "ephemeral"
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "true"

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api.app import app


TEST_API_KEY = "test-secret-key"


@pytest.fixture
def sample_prediction_payload():
    return {
        "inputs": [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Date": "2026-03-06",
                "Customers": 500,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            }
        ]
    }


@pytest.fixture
def sample_prediction_df():
    return pd.DataFrame(
        [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Date": "2026-03-06",
                "Customers": 500,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            }
        ]
    )

@pytest.fixture
def sample_store_metadata():
    return pd.DataFrame(
        [
            {
                "Store": 1,
                "StoreType": "a",
                "Assortment": "a",
                "CompetitionDistance": 500.0,
                "Promo2": 1,
            }
        ]
    )


@pytest.fixture
def sample_store_state():
    return {
        "1": [1000, 1100, 1200, 1300, 1400, 1500, 1600]
    }


@pytest.fixture
def mock_xgb_model():
    model = MagicMock()
    model.get_booster.return_value.feature_names = [
        "Store",
        "DayOfWeek",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "Promo2",
        "WeekOfYear",
        "day",
        "month",
        "year",
        "is_month_start",
        "is_month_end",
        "sales_lag_1",
        "sales_lag_7",
        "sales_rolling_mean_7",
    ]
    model.predict.return_value = [8.597486]
    return model


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture
def api_headers():
    return {"X-API-KEY": TEST_API_KEY}