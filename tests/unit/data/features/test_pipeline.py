from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import src.data.features.pipeline as pipeline
from src.data.features.pipeline import (
    load_validated_inputs,
    merge_feature_sources,
    persist_feature_dataset,
    resolve_entity_column,
    run_feature_pipeline,
)


def test_resolve_entity_column_returns_first_configured_id():
    config = {
        "data": {
            "id_columns": [
                "Store",
                "Region",
            ],
        }
    }

    result = resolve_entity_column(
        config
    )

    assert result == "Store"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            "data": {},
        },
        {
            "data": {
                "id_columns": [],
            },
        },
    ],
)
def test_resolve_entity_column_rejects_missing_ids(
    config: dict,
):
    with pytest.raises(
        ValueError,
        match="data.id_columns",
    ):
        resolve_entity_column(
            config
        )


def test_load_validated_inputs_rejects_missing_train(
    tmp_path,
):
    store = pd.DataFrame(
        {
            "Store": [1],
        }
    )

    store.to_parquet(
        tmp_path / "store.parquet",
        index=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Validated training or store data is missing",
    ):
        load_validated_inputs(
            validated_path=str(tmp_path),
        )


def test_load_validated_inputs_rejects_missing_store(
    tmp_path,
):
    train = pd.DataFrame(
        {
            "Store": [1],
            "Sales": [1000.0],
        }
    )

    train.to_parquet(
        tmp_path / "train.parquet",
        index=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Validated training or store data is missing",
    ):
        load_validated_inputs(
            validated_path=str(tmp_path),
        )


def test_load_validated_inputs_loads_train_and_store(
    tmp_path,
):
    train = pd.DataFrame(
        {
            "Store": [1, 2],
            "Sales": [
                1000.0,
                900.0,
            ],
        }
    )
    store = pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
        }
    )

    train.to_parquet(
        tmp_path / "train.parquet",
        index=False,
    )
    store.to_parquet(
        tmp_path / "store.parquet",
        index=False,
    )

    datasets = load_validated_inputs(
        validated_path=str(tmp_path),
    )

    assert set(datasets) == {
        "train",
        "store",
    }

    pd.testing.assert_frame_equal(
        datasets["train"],
        train,
    )
    pd.testing.assert_frame_equal(
        datasets["store"],
        store,
    )


def test_merge_feature_sources_uses_configured_entity_key(
    monkeypatch,
):
    train = pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                ]
            ),
            "Sales": [
                1000.0,
                900.0,
            ],
        }
    )
    store = pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
        }
    )
    calendar = pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                ]
            ),
        }
    )

    expected = train.merge(
        store,
        on="Store",
        how="left",
    ).assign(
        calendar_attached=True
    )

    load_calendar_mock = Mock(
        return_value=calendar
    )
    merge_calendar_mock = Mock(
        return_value=expected
    )

    monkeypatch.setattr(
        pipeline,
        "load_known_calendar",
        load_calendar_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "merge_known_calendar_features",
        merge_calendar_mock,
    )

    result = merge_feature_sources(
        {
            "train": train,
            "store": store,
        },
        config={
            "data": {
                "id_columns": [
                    "Store",
                ],
            }
        },
    )

    assert "StoreType" in result.columns
    assert result["calendar_attached"].all()

    load_calendar_mock.assert_called_once_with()

    merged_before_calendar = (
        merge_calendar_mock.call_args.args[0]
    )

    assert "StoreType" in (
        merged_before_calendar.columns
    )

    merge_calendar_mock.assert_called_once_with(
        merged_before_calendar,
        calendar,
        strict=True,
    )


def test_merge_feature_sources_rejects_missing_dataset():
    with pytest.raises(
        KeyError,
    ):
        merge_feature_sources(
            {
                "train": pd.DataFrame(
                    {
                        "Store": [1],
                    }
                ),
            },
            config={
                "data": {
                    "id_columns": [
                        "Store",
                    ],
                }
            },
        )


def test_persist_feature_dataset_writes_parquet(
    tmp_path,
):
    features = pd.DataFrame(
        {
            "Store": [1, 2],
            "sales_lag_1": [
                0.0,
                1000.0,
            ],
        }
    )

    output_path = persist_feature_dataset(
        features,
        features_path=str(tmp_path),
    )

    expected_path = (
        tmp_path
        / "features.parquet"
    )

    assert Path(output_path) == expected_path
    assert expected_path.exists()

    saved = pd.read_parquet(
        expected_path
    )

    pd.testing.assert_frame_equal(
        saved,
        features,
    )


def test_run_feature_pipeline_executes_all_steps(
    monkeypatch,
):
    config = {
        "data": {
            "id_columns": [
                "Store",
            ],
        }
    }

    datasets = {
        "train": pd.DataFrame(
            {
                "Store": [1],
            }
        ),
        "store": pd.DataFrame(
            {
                "Store": [1],
            }
        ),
    }

    merged = pd.DataFrame(
        {
            "Store": [1],
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
        }
    )

    features = merged.assign(
        sales_lag_1=0.0
    )

    load_mock = Mock(
        return_value=datasets
    )
    merge_mock = Mock(
        return_value=merged
    )
    build_mock = Mock(
        return_value=features
    )
    persist_mock = Mock(
        return_value=(
            "data/features/features.parquet"
        )
    )

    monkeypatch.setattr(
        pipeline,
        "load_validated_inputs",
        load_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "merge_feature_sources",
        merge_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "build_features",
        build_mock,
    )
    monkeypatch.setattr(
        pipeline,
        "persist_feature_dataset",
        persist_mock,
    )

    run_feature_pipeline(
        config=config
    )

    load_mock.assert_called_once_with()

    merge_mock.assert_called_once_with(
        datasets,
        config=config,
    )

    build_mock.assert_called_once_with(
        merged,
        config=config,
        mode="train",
    )

    persist_mock.assert_called_once_with(
        features
    )