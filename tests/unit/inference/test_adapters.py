import pandas as pd
import pytest

from mlops_sales_forecasting.inference.adapters import (
    request_to_dataframe,
    resolve_forecasting_store_id,
    resolve_open_flags,
)


def test_request_to_dataframe() -> None:
    inputs = [
        {
            "Store": 1,
            "Open": 1,
            "Promo": 1,
        },
        {
            "Store": 2,
            "Open": 0,
            "Promo": 0,
        },
    ]

    result = request_to_dataframe(inputs)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == [
        "Store",
        "Open",
        "Promo",
    ]


def test_request_to_dataframe_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="No input rows provided",
    ):
        request_to_dataframe([])


def test_resolve_forecasting_store_id() -> None:
    frame = pd.DataFrame(
        [
            {
                "Store": 42,
            },
        ]
    )

    assert resolve_forecasting_store_id(frame) == 42


def test_resolve_forecasting_store_id_requires_store() -> None:
    frame = pd.DataFrame(
        [
            {
                "Promo": 1,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="requires field 'Store'",
    ):
        resolve_forecasting_store_id(frame)


def test_resolve_open_flags() -> None:
    frame = pd.DataFrame(
        [
            {
                "Open": 1,
            },
            {
                "Open": 0,
            },
        ]
    )

    assert resolve_open_flags(frame) == [1, 0]


def test_resolve_open_flags_returns_none_when_missing() -> None:
    frame = pd.DataFrame(
        [
            {
                "Promo": 1,
            },
        ]
    )

    assert resolve_open_flags(frame) is None