import json

import pandas as pd

from mlops_sales_forecasting.data.features.create_state import build_feature_state
from mlops_sales_forecasting.data.features.update_state import (
    update_feature_state_from_ground_truth,
)


def test_persisted_dated_updates_are_idempotent(tmp_path) -> None:
    config = {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": ["Store"],
        },
        "features": {
            "lag_features": {
                "lags": [1, 3],
                "rolling_windows": [3],
            },
        },
    }

    state_path = tmp_path / "state.json"
    batch_path = tmp_path / "batch.csv"

    initial = build_feature_state(
        pd.DataFrame(
            {
                "Store": [1, 1],
                "Date": ["2026-01-01", "2026-01-03"],
                "Sales": [10, 30],
            }
        ),
        config,
    )
    state_path.write_text(json.dumps(initial), encoding="utf-8")

    # Includes a late observation, a correction and a new day.
    pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": ["2026-01-02", "2026-01-03", "2026-01-04"],
            "Sales": [20, 35, 40],
        }
    ).to_csv(batch_path, index=False)

    first = update_feature_state_from_ground_truth(
        str(batch_path),
        config=config,
        state_path=str(state_path),
    )
    first_state = json.loads(state_path.read_text(encoding="utf-8"))

    second = update_feature_state_from_ground_truth(
        str(batch_path),
        config=config,
        state_path=str(state_path),
    )
    second_state = json.loads(state_path.read_text(encoding="utf-8"))

    assert first_state["stores"]["1"] == [
        {"date": "2026-01-02", "value": 20.0},
        {"date": "2026-01-03", "value": 35.0},
        {"date": "2026-01-04", "value": 40.0},
    ]
    assert first_state == second_state
    assert first["updated_entities"] == 1
    assert first["appended_values"] == 2
    assert second["updated_entities"] == 0
    assert second["appended_values"] == 0