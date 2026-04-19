import pandas as pd
import pytest

from src.inference.adapters import (
    request_to_dataframe,
    resolve_forecasting_store_id,
    resolve_open_flags,
)


def test_request_to_dataframe_success():
    inputs = [
        {"Store": 1, "Open": 1, "Promo": 1},
        {"Store": 2, "Open": 0, "Promo": 0},
    ]

    df = request_to_dataframe(inputs)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["Store", "Open", "Promo"]


def test_request_to_dataframe_empty():
    with pytest.raises(ValueError, match="No input rows provided"):
        request_to_dataframe([])


def test_resolve_forecasting_store_id_success():
    df = pd.DataFrame([{"Store": 42}])

    store_id = resolve_forecasting_store_id(df)

    assert store_id == 42


def test_resolve_forecasting_store_id_missing():
    df = pd.DataFrame([{"Promo": 1}])

    with pytest.raises(ValueError, match="requires field 'Store'"):
        resolve_forecasting_store_id(df)


def test_resolve_open_flags_present():
    df = pd.DataFrame([{"Open": 1}, {"Open": 0}])

    open_flags = resolve_open_flags(df)

    assert open_flags == [1, 0]


def test_resolve_open_flags_missing():
    df = pd.DataFrame([{"Promo": 1}])

    open_flags = resolve_open_flags(df)

    assert open_flags is None