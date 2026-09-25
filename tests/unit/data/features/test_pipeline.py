from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.data.contracts import (
    DatasetCollection,
)
from mlops_sales_forecasting.data.features import pipeline
from mlops_sales_forecasting.data.features.pipeline import (
    load_feature_inputs,
    persist_feature_dataset,
    run_feature_pipeline,
)


def test_load_feature_inputs_rejects_missing_files(
    tmp_path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Required feature inputs are missing",
    ):
        load_feature_inputs(
            validated_path=str(tmp_path),
            calendar_path=str(tmp_path / "known_calendar.parquet"),
        )


def test_persist_feature_dataset_writes_parquet(
    tmp_path,
) -> None:
    features = pd.DataFrame(
        {
            "Store": [
                1,
            ],
            "sales_lag_1": [
                0.0,
            ],
        }
    )

    result = persist_feature_dataset(
        features,
        features_path=str(tmp_path),
    )

    assert result == str(tmp_path / "features.parquet")
    assert (tmp_path / "features.parquet").is_file()


def test_run_feature_pipeline_executes_all_steps(
    tmp_path,
    monkeypatch,
) -> None:
    config = {
        "paths": {
            "raw_data": str(tmp_path / "raw"),
            "validated_data": str(tmp_path / "validation"),
            "features": str(tmp_path / "features"),
            "models": str(tmp_path / "models"),
        },
    }
    datasets = DatasetCollection(
        datasets={
            "train": pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
            "store": pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
            "known_calendar": pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
        }
    )
    features = pd.DataFrame(
        {
            "Store": [
                1,
            ],
            "sales_lag_1": [
                0.0,
            ],
        }
    )

    create_calendar = MagicMock()
    load_inputs = MagicMock(return_value=datasets)
    build_features = MagicMock(return_value=features)
    persist_features = MagicMock(return_value=str(tmp_path / "features" / "features.parquet"))
    create_state = MagicMock()

    monkeypatch.setattr(
        pipeline,
        "create_known_calendar_artifact",
        create_calendar,
    )
    monkeypatch.setattr(
        pipeline,
        "load_feature_inputs",
        load_inputs,
    )
    monkeypatch.setattr(
        pipeline.RossmannFeatureBuilder,
        "build_features",
        build_features,
    )
    monkeypatch.setattr(
        pipeline,
        "persist_feature_dataset",
        persist_features,
    )
    monkeypatch.setattr(
        pipeline,
        "create_feature_state",
        create_state,
    )

    result = run_feature_pipeline(config)

    assert result["feature_rows"] == 1
    assert result["state_path"] == str(tmp_path / "models" / "latest_state.json")

    create_calendar.assert_called_once()
    load_inputs.assert_called_once()
    build_features.assert_called_once()
    persist_features.assert_called_once()
    create_state.assert_called_once()
