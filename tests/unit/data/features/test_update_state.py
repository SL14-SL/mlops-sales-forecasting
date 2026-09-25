import json

import pandas as pd
import pytest

from mlops_sales_forecasting.data.features.update_state import (
    update_feature_state_from_ground_truth,
)


def forecasting_config() -> dict:
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
                    2,
                ],
            },
        },
    }


def test_update_feature_state_appends_ground_truth(
    tmp_path,
) -> None:
    state_path = tmp_path / "models" / "latest_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "1": [
                    10.0,
                    20.0,
                    30.0,
                ],
                "2": [
                    50.0,
                ],
            }
        ),
        encoding="utf-8",
    )

    batch_path = tmp_path / "ground_truth.csv"
    pd.DataFrame(
        {
            "Store": [
                1,
                1,
                3,
            ],
            "Date": [
                "2026-01-04",
                "2026-01-05",
                "2026-01-01",
            ],
            "Sales": [
                40.0,
                45.0,
                70.0,
            ],
        }
    ).to_csv(
        batch_path,
        index=False,
    )

    result = update_feature_state_from_ground_truth(
        str(batch_path),
        config=forecasting_config(),
        state_path=str(state_path),
    )

    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert persisted == {
        "1": [
            30.0,
            40.0,
            45.0,
        ],
        "2": [
            50.0,
        ],
        "3": [
            70.0,
        ],
    }
    assert result == {
        "state_path": str(state_path),
        "history_length": 3,
        "updated_entities": 2,
        "appended_values": 3,
        "state_entities": 3,
    }


def test_update_feature_state_keeps_last_duplicate(
    tmp_path,
) -> None:
    state_path = tmp_path / "latest_state.json"
    batch_path = tmp_path / "ground_truth.csv"

    pd.DataFrame(
        {
            "Store": [
                1,
                1,
            ],
            "Date": [
                "2026-01-01",
                "2026-01-01",
            ],
            "Sales": [
                100.0,
                120.0,
            ],
        }
    ).to_csv(
        batch_path,
        index=False,
    )

    update_feature_state_from_ground_truth(
        str(batch_path),
        config=forecasting_config(),
        state_path=str(state_path),
    )

    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert persisted == {
        "1": [
            120.0,
        ],
    }


def test_update_feature_state_rejects_missing_batch(
    tmp_path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Ground-truth batch not found",
    ):
        update_feature_state_from_ground_truth(
            str(tmp_path / "missing.csv"),
            config=forecasting_config(),
            state_path=str(tmp_path / "state.json"),
        )
