from typing import Any

import pandas as pd
import pytest

from mlops_sales_forecasting.inference import (
    forecasting_provider,
)
from mlops_sales_forecasting.inference.forecasting_provider import (
    build_forecasting_inference_features,
)


@pytest.fixture
def config() -> dict[str, Any]:
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
                    7,
                ],
                "rolling_windows": [
                    7,
                ],
            },
        },
    }


def test_build_forecasting_inference_features(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
) -> None:
    calls: list[str] = []

    validated_frame = pd.DataFrame(
        [
            {
                "Store": 1,
                "Date": pd.Timestamp("2026-09-25"),
            },
        ]
    )
    store_metadata = pd.DataFrame(
        [
            {
                "Store": 1,
                "StoreType": "a",
            },
        ]
    )
    known_calendar = pd.DataFrame(
        [
            {
                "Store": 1,
                "Date": pd.Timestamp("2026-09-25"),
            },
        ]
    )
    store_state = {
        "1": [
            100.0,
        ],
    }

    def normalize(
        frame: pd.DataFrame,
        received_config: dict,
    ) -> pd.DataFrame:
        assert frame is validated_frame
        assert received_config is config
        calls.append("normalize")
        return frame.assign(normalized=True)

    def merge_metadata(
        frame: pd.DataFrame,
        metadata: pd.DataFrame,
        received_config: dict,
    ) -> pd.DataFrame:
        assert frame["normalized"].all()
        assert metadata is store_metadata
        assert received_config is config
        calls.append("metadata")
        return frame.assign(metadata=True)

    def merge_calendar(
        frame: pd.DataFrame,
        calendar: pd.DataFrame,
        received_config: dict,
    ) -> pd.DataFrame:
        assert frame["metadata"].all()
        assert calendar is known_calendar
        assert received_config is config
        calls.append("calendar")
        return frame.assign(calendar=True)

    def engineer(
        frame: pd.DataFrame,
        received_config: dict,
    ) -> pd.DataFrame:
        assert frame["calendar"].all()
        assert received_config is config
        calls.append("features")
        return frame.assign(engineered=True)

    def inject_state(
        frame: pd.DataFrame,
        state: dict,
        received_config: dict,
    ) -> pd.DataFrame:
        assert frame["engineered"].all()
        assert state is store_state
        assert received_config is config
        calls.append("state")
        return frame.assign(state=True)

    def finalize(
        frame: pd.DataFrame,
        received_config: dict,
    ) -> pd.DataFrame:
        assert frame["state"].all()
        assert received_config is config
        calls.append("finalize")
        return frame.drop(
            columns="Date",
        )

    monkeypatch.setattr(
        forecasting_provider,
        "normalize_store_key",
        normalize,
    )
    monkeypatch.setattr(
        forecasting_provider,
        "merge_request_with_metadata",
        merge_metadata,
    )
    monkeypatch.setattr(
        forecasting_provider,
        "merge_request_with_calendar",
        merge_calendar,
    )
    monkeypatch.setattr(
        forecasting_provider,
        "run_forecasting_feature_engineering",
        engineer,
    )
    monkeypatch.setattr(
        forecasting_provider,
        "inject_forecasting_state_features",
        inject_state,
    )
    monkeypatch.setattr(
        forecasting_provider,
        "finalize_forecasting_feature_frame",
        finalize,
    )

    result = build_forecasting_inference_features(
        validated_frame=validated_frame,
        store_metadata=store_metadata,
        store_state=store_state,
        known_calendar=known_calendar,
        config=config,
    )

    assert calls == [
        "normalize",
        "metadata",
        "calendar",
        "features",
        "state",
        "finalize",
    ]
    assert "Date" not in result.columns
    assert result["state"].all()


def test_build_forecasting_features_requires_store(
    config: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError,
        match="requires entity column 'Store'",
    ):
        build_forecasting_inference_features(
            validated_frame=pd.DataFrame(
                [
                    {
                        "Promo": 1,
                    },
                ]
            ),
            store_metadata=pd.DataFrame(
                [
                    {
                        "Store": 1,
                    },
                ]
            ),
            store_state={},
            known_calendar=pd.DataFrame(),
            config=config,
        )