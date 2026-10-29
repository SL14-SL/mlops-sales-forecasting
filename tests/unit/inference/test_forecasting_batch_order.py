from types import SimpleNamespace

import pandas as pd
import pytest

from mlops_sales_forecasting.api.schema import PredictionRequest
from mlops_sales_forecasting.data.features.calendar import CALENDAR_FEATURE_COLUMNS
from mlops_sales_forecasting.inference.forecasting_provider import (
    build_forecasting_inference_features,
)
from mlops_sales_forecasting.inference.prediction_service import predict_with_bundle


class StoreIdentityModel:
    """Return a distinct, predictable value for each store."""

    input_columns = ["Store"]

    def predict(self, frame: pd.DataFrame):
        return frame["Store"].to_numpy(dtype=float) * 100.0


@pytest.mark.parametrize(
    "rows",
    [
        [(2, 1), (1, 0)],
        [(2, 0), (1, 1)],
        [(2, 1), (1, 1)],
        [(1, 0), (1, 1)],
    ],
    ids=[
        "unsorted-first-open",
        "unsorted-first-closed",
        "unsorted-both-open",
        "duplicate-store-and-date",
    ],
)
def test_forecasting_preserves_request_order(
    rows: list[tuple[int, int]],
) -> None:
    inputs = [
        {
            "Store": store,
            "Date": "2026-09-25",
            "Open": is_open,
            "Promo": 0,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }
        for store, is_open in rows
    ]

    config = {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": ["Store"],
        },
        "features": {
            "enabled_steps": ["sort", "lags"],
            "lag_features": {
                "lags": [1],
                "rolling_windows": [1],
            },
        },
    }

    # Duplicate dataframe indexes must not affect row identity.
    frame = pd.DataFrame(inputs, index=[9, 9])
    frame["Date"] = pd.to_datetime(frame["Date"])

    calendar = (
        frame[["Store", "Date"]]
        .drop_duplicates()
        .assign(**dict.fromkeys(CALENDAR_FEATURE_COLUMNS, 0))
    )

    bundle = SimpleNamespace(
        model=StoreIdentityModel(),
        release_id="batch-order-test",
        target_transformation="none",
        store_metadata=pd.DataFrame({"Store": [1, 2]}),
        store_state={"1": [100.0], "2": [200.0]},
        known_calendar=calendar,
    )

    features = build_forecasting_inference_features(
        validated_frame=frame,
        store_metadata=bundle.store_metadata,
        store_state=bundle.store_state,
        known_calendar=bundle.known_calendar,
        config=config,
    )

    assert features["Store"].tolist() == [store for store, _ in rows]
    assert features["Open"].tolist() == [is_open for _, is_open in rows]
    assert features["sales_lag_1"].tolist() == [
        store * 100.0 for store, _ in rows
    ]
    assert features.index.tolist() == [0, 1]
    assert "__request_row_order" not in features.columns

    response = predict_with_bundle(
        PredictionRequest(inputs=inputs),
        bundle,
        config,
    )

    assert [result.row_index for result in response.predictions] == [0, 1]
    assert [result.prediction for result in response.predictions] == [
        store * 100.0 if is_open else 0.0
        for store, is_open in rows
    ]