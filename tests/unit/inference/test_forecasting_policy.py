import pandas as pd
import pytest

from mlops_sales_forecasting.inference.forecasting_policy import (
    apply_forecasting_business_rules,
    finalize_forecasting_feature_frame,
    inject_forecasting_state_features,
    merge_request_with_metadata,
    normalize_store_key,
)


@pytest.fixture
def config() -> dict:
    return {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": [
                "Store",
            ],
        },
        "features": {
            "lag_features": {
                "lags": [
                    1,
                    3,
                ],
                "rolling_windows": [
                    3,
                ],
            },
        },
    }


def test_normalize_store_key(
    config: dict,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "Store": "7",
            },
        ]
    )

    result = normalize_store_key(
        frame,
        config,
    )

    assert result["Store"].tolist() == [7]
    assert result["Store"].dtype == int


def test_normalize_store_key_requires_entity_column(
    config: dict,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires entity column 'Store'",
    ):
        normalize_store_key(
            pd.DataFrame(
                [
                    {
                        "Promo": 1,
                    },
                ]
            ),
            config,
        )


def test_merge_request_with_metadata(
    config: dict,
) -> None:
    request = pd.DataFrame(
        [
            {
                "Store": 2,
                "Promo": 1,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "Store": 2,
                "StoreType": "a",
                "Promo2": 1,
            },
        ]
    )

    result = merge_request_with_metadata(
        request,
        metadata,
        config,
    )

    assert result.loc[0, "StoreType"] == "a"
    assert result.loc[0, "Promo2"] == 1


def test_merge_request_rejects_missing_metadata(
    config: dict,
) -> None:
    request = pd.DataFrame(
        [
            {
                "Store": 3,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "Store": 2,
                "StoreType": "a",
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"Store values: \[3\]",
    ):
        merge_request_with_metadata(
            request,
            metadata,
            config,
        )


def test_inject_forecasting_state_features(
    config: dict,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "Store": 1,
                "sales_lag_1": 0.0,
                "sales_lag_3": 0.0,
                "sales_rolling_mean_3": 0.0,
            },
            {
                "Store": 2,
                "sales_lag_1": 0.0,
                "sales_lag_3": 0.0,
                "sales_rolling_mean_3": 0.0,
            },
        ]
    )
    state = {
        "1": [
            10.0,
            20.0,
            30.0,
        ],
        "2": [
            50.0,
        ],
    }

    result = inject_forecasting_state_features(
        frame,
        state,
        config,
    )

    assert result.loc[0, "sales_lag_1"] == 30.0
    assert result.loc[0, "sales_lag_3"] == 10.0
    assert result.loc[
        0,
        "sales_rolling_mean_3",
    ] == 20.0

    assert result.loc[1, "sales_lag_1"] == 50.0
    assert result.loc[1, "sales_lag_3"] == 0.0
    assert result.loc[
        1,
        "sales_rolling_mean_3",
    ] == 0.0


def test_finalize_feature_frame_drops_date(
    config: dict,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "Store": 1,
                "Date": pd.Timestamp("2026-09-25"),
                "Promo": 1,
            },
        ]
    )

    result = finalize_forecasting_feature_frame(
        frame,
        config,
    )

    assert "Date" not in result.columns
    assert result.loc[0, "Store"] == 1


@pytest.mark.parametrize(
    ("prediction", "is_open", "expected"),
    [
        (1234.5, 1, 1234.5),
        (1234.5, 0, 0.0),
        (-10.0, 1, 0.0),
    ],
)
def test_apply_forecasting_business_rules(
    prediction: float,
    is_open: int,
    expected: float,
) -> None:
    assert (
        apply_forecasting_business_rules(
            prediction,
            is_open,
        )
        == expected
    )