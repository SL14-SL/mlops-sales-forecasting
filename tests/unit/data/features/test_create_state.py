import json

import pandas as pd

from mlops_sales_forecasting.data.features.create_state import (
    build_feature_state,
    create_feature_state,
)


def forecasting_config() -> dict:
    return {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": ["Store"],
        },
        "features": {
            "lag_features": {
                "lags": [1, 3],
                "rolling_windows": [2],
            },
        },
    }


def test_build_feature_state_keeps_required_dated_history() -> None:
    features = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1, 2, 2],
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "Sales": [10, 20, 30, 40, 50, 60],
        }
    )

    result = build_feature_state(features, forecasting_config())

    assert result == {
        "schema_version": 1,
        "stores": {
            "1": [
                {"date": "2026-01-02", "value": 20.0},
                {"date": "2026-01-03", "value": 30.0},
                {"date": "2026-01-04", "value": 40.0},
            ],
            "2": [
                {"date": "2026-01-01", "value": 50.0},
                {"date": "2026-01-02", "value": 60.0},
            ],
        },
    }


def test_create_feature_state_persists_dated_json(tmp_path) -> None:
    features_path = tmp_path / "features.parquet"
    state_path = tmp_path / "models" / "latest_state.json"

    features = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Sales": [100, 120],
        }
    )
    features.to_parquet(features_path, index=False)

    result = create_feature_state(
        config=forecasting_config(),
        features_path=str(features_path),
        state_path=str(state_path),
    )

    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert result == build_feature_state(features, forecasting_config())
    assert persisted == result
    assert persisted["schema_version"] == 1