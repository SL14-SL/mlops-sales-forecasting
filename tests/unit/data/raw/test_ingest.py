import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data.raw.ingest import (
    collect_incremental_batches,
    collect_local_batches,
    collect_remote_batches,
    create_simulation_split,
    ingest,
    load_base_datasets,
    merge_training_batches,
    persist_simulation_source_if_missing,
    persist_validated_datasets,
)


def build_training_frame(
    dates: list[str],
    *,
    sales: list[float] | None = None,
) -> pd.DataFrame:
    """Build a schema-valid forecasting training dataframe."""
    row_count = len(dates)

    if sales is None:
        sales = [
            float(1000 + index * 100)
            for index in range(row_count)
        ]

    return pd.DataFrame(
        {
            "Store": [1] * row_count,
            "DayOfWeek": [
                index % 7 + 1
                for index in range(row_count)
            ],
            "Date": dates,
            "Sales": sales,
            "Customers": [100] * row_count,
            "Open": [1] * row_count,
            "Promo": [0] * row_count,
            "StateHoliday": ["0"] * row_count,
            "SchoolHoliday": [0] * row_count,
        }
    )


def build_store_frame() -> pd.DataFrame:
    """Build schema-valid store metadata."""
    return pd.DataFrame(
        {
            "Store": [1],
            "StoreType": ["a"],
            "Assortment": ["a"],
            "CompetitionDistance": [500.0],
        }
    )


def test_create_simulation_split_is_chronological():
    frame = build_training_frame(
        [
            "2026-01-03",
            "2026-01-01",
            "2026-01-04",
            "2026-01-02",
        ]
    )

    train, simulation = create_simulation_split(
        frame,
        training_fraction=0.5,
    )

    assert not train.empty
    assert not simulation.empty

    assert (
        train["Date"].max()
        < simulation["Date"].min()
    )

    assert train["Date"].is_monotonic_increasing
    assert simulation["Date"].is_monotonic_increasing


def test_create_simulation_split_keeps_same_date_together():
    frame = pd.DataFrame(
        {
            "Store": [1, 2, 1, 2],
            "DayOfWeek": [1, 1, 2, 2],
            "Date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
            ],
            "Sales": [
                1000.0,
                900.0,
                1100.0,
                950.0,
            ],
            "Customers": [100, 90, 110, 95],
            "Open": [1, 1, 1, 1],
            "Promo": [0, 0, 0, 0],
            "StateHoliday": ["0", "0", "0", "0"],
            "SchoolHoliday": [0, 0, 0, 0],
        }
    )

    train, simulation = create_simulation_split(
        frame,
        training_fraction=0.5,
    )

    train_dates = set(
        train["Date"]
    )
    simulation_dates = set(
        simulation["Date"]
    )

    assert train_dates.isdisjoint(
        simulation_dates
    )
    assert len(train) == 2
    assert len(simulation) == 2


def test_create_simulation_split_rejects_single_date():
    frame = build_training_frame(
        [
            "2026-01-01",
            "2026-01-01",
        ]
    )

    with pytest.raises(
        ValueError,
        match="At least two unique dates",
    ):
        create_simulation_split(
            frame
        )


@pytest.mark.parametrize(
    "training_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_create_simulation_split_rejects_invalid_fraction(
    training_fraction: float,
):
    frame = build_training_frame(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    with pytest.raises(
        ValueError,
        match="training_fraction",
    ):
        create_simulation_split(
            frame,
            training_fraction=training_fraction,
        )


def test_create_simulation_split_rejects_invalid_dates():
    frame = build_training_frame(
        [
            "2026-01-01",
            "not-a-date",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Found invalid dates",
    ):
        create_simulation_split(
            frame
        )


def test_merge_training_batches_combines_and_sorts():
    train_base = build_training_frame(
        [
            "2026-01-01",
        ],
        sales=[
            1000.0,
        ],
    )

    batch = build_training_frame(
        [
            "2026-01-03",
            "2026-01-02",
        ],
        sales=[
            1200.0,
            1100.0,
        ],
    )

    merged = merge_training_batches(
        train_base,
        [
            batch,
        ],
    )

    assert len(merged) == 3
    assert merged["Date"].is_monotonic_increasing
    assert merged["Sales"].tolist() == [
        1000.0,
        1100.0,
        1200.0,
    ]


def test_merge_training_batches_without_batches_returns_copy():
    train_base = build_training_frame(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    merged = merge_training_batches(
        train_base,
        [],
    )

    pd.testing.assert_frame_equal(
        merged,
        train_base,
    )

    assert merged is not train_base


def test_persist_simulation_source_creates_missing_file(
    tmp_path,
):
    simulation = build_training_frame(
        [
            "2026-01-02",
        ]
    )

    persist_simulation_source_if_missing(
        simulation,
        raw_path=str(tmp_path),
    )

    output_path = (
        tmp_path
        / "simulation_ground_truth.csv"
    )

    assert output_path.exists()

    saved = pd.read_csv(
        output_path
    )

    assert len(saved) == 1
    assert saved.loc[0, "Sales"] == 1000.0


def test_persist_simulation_source_preserves_existing_file(
    tmp_path,
):
    output_path = (
        tmp_path
        / "simulation_ground_truth.csv"
    )

    output_path.write_text(
        "existing-content",
        encoding="utf-8",
    )

    simulation = build_training_frame(
        [
            "2026-01-02",
        ]
    )

    persist_simulation_source_if_missing(
        simulation,
        raw_path=str(tmp_path),
    )

    assert output_path.read_text(
        encoding="utf-8"
    ) == "existing-content"


def test_collect_local_batches_returns_valid_batches(
    tmp_path,
):
    batch_directory = (
        tmp_path
        / "new_batches"
    )
    quarantine_directory = (
        tmp_path
        / "quarantine"
    )

    batch_directory.mkdir()

    valid_batch = build_training_frame(
        [
            "2026-01-03",
        ]
    )

    valid_batch.to_csv(
        batch_directory / "valid.csv",
        index=False,
    )

    batches = collect_local_batches(
        batch_directory=str(
            batch_directory
        ),
        quarantine_directory=str(
            quarantine_directory
        ),
    )

    assert len(batches) == 1
    assert len(batches[0]) == 1

    assert (
        batch_directory
        / "valid.csv"
    ).exists()

    assert not (
        quarantine_directory
        / "valid.csv"
    ).exists()


def test_collect_local_batches_moves_invalid_batch_to_quarantine(
    tmp_path,
):
    batch_directory = (
        tmp_path
        / "new_batches"
    )
    quarantine_directory = (
        tmp_path
        / "quarantine"
    )

    batch_directory.mkdir()

    invalid_batch = build_training_frame(
        [
            "2026-01-03",
        ],
        sales=[
            -99.0,
        ],
    )

    invalid_path = (
        batch_directory
        / "invalid.csv"
    )

    invalid_batch.to_csv(
        invalid_path,
        index=False,
    )

    batches = collect_local_batches(
        batch_directory=str(
            batch_directory
        ),
        quarantine_directory=str(
            quarantine_directory
        ),
    )

    assert batches == []
    assert not invalid_path.exists()

    assert (
        quarantine_directory
        / "invalid.csv"
    ).exists()


def test_collect_local_batches_ignores_non_csv_files(
    tmp_path,
):
    batch_directory = (
        tmp_path
        / "new_batches"
    )
    quarantine_directory = (
        tmp_path
        / "quarantine"
    )

    batch_directory.mkdir()

    ignored_file = (
        batch_directory
        / "notes.txt"
    )

    ignored_file.write_text(
        "not a batch",
        encoding="utf-8",
    )

    batches = collect_local_batches(
        batch_directory=str(
            batch_directory
        ),
        quarantine_directory=str(
            quarantine_directory
        ),
    )

    assert batches == []
    assert ignored_file.exists()


def test_collect_local_batches_returns_empty_for_missing_directory(
    tmp_path,
):
    batches = collect_local_batches(
        batch_directory=str(
            tmp_path
            / "missing"
        ),
        quarantine_directory=str(
            tmp_path
            / "quarantine"
        ),
    )

    assert batches == []


def test_collect_remote_batches_ignores_invalid_batch(
    monkeypatch,
):
    valid_batch = build_training_frame(
        [
            "2026-01-03",
        ]
    )

    list_files_mock = Mock(
        return_value=[
            "gs://bucket/new_batches/valid.csv",
            "gs://bucket/new_batches/invalid.csv",
        ]
    )

    load_mock = Mock(
        side_effect=[
            valid_batch,
            ValueError(
                "invalid batch"
            ),
        ]
    )

    monkeypatch.setattr(
        "src.data.raw.ingest.list_files",
        list_files_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.load_and_validate_batch",
        load_mock,
    )

    batches = collect_remote_batches(
        batch_directory=(
            "gs://bucket/new_batches"
        ),
    )

    assert len(batches) == 1

    pd.testing.assert_frame_equal(
        batches[0],
        valid_batch,
    )

    list_files_mock.assert_called_once_with(
        "gs://bucket/new_batches/*.csv"
    )


@pytest.mark.parametrize(
    (
        "environment",
        "raw_path",
        "expected_collector",
    ),
    [
        (
            "prod",
            "data/raw",
            "remote",
        ),
        (
            "dev",
            "gs://bucket/data/raw",
            "remote",
        ),
        (
            "dev",
            "data/raw",
            "local",
        ),
    ],
)
def test_collect_incremental_batches_selects_storage_mode(
    monkeypatch,
    environment: str,
    raw_path: str,
    expected_collector: str,
):
    remote_mock = Mock(
        return_value=[]
    )
    local_mock = Mock(
        return_value=[]
    )

    monkeypatch.setattr(
        "src.data.raw.ingest.collect_remote_batches",
        remote_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.collect_local_batches",
        local_mock,
    )

    collect_incremental_batches(
        raw_path=raw_path,
        environment=environment,
    )

    if expected_collector == "remote":
        remote_mock.assert_called_once()
        local_mock.assert_not_called()
    else:
        local_mock.assert_called_once()
        remote_mock.assert_not_called()


def test_load_base_datasets_loads_and_validates_sources(
    monkeypatch,
):
    train = build_training_frame(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )
    store = build_store_frame()

    read_csv_mock = Mock(
        side_effect=[
            train,
            store,
        ]
    )
    validate_train_mock = Mock(
        return_value=train
    )
    validate_store_mock = Mock(
        return_value=store
    )

    monkeypatch.setattr(
        "src.data.raw.ingest.pd.read_csv",
        read_csv_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.validate_train",
        validate_train_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.validate_store",
        validate_store_mock,
    )

    loaded_train, loaded_store = (
        load_base_datasets(
            "data/raw"
        )
    )

    pd.testing.assert_frame_equal(
        loaded_train,
        train,
    )
    pd.testing.assert_frame_equal(
        loaded_store,
        store,
    )

    validate_train_mock.assert_called_once_with(
        train
    )
    validate_store_mock.assert_called_once_with(
        store
    )


def test_persist_validated_datasets_writes_parquet(
    tmp_path,
):
    train = build_training_frame(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )
    store = build_store_frame()

    persist_validated_datasets(
        train,
        store,
        validated_path=str(tmp_path),
    )

    train_path = (
        tmp_path
        / "train.parquet"
    )
    store_path = (
        tmp_path
        / "store.parquet"
    )

    assert train_path.exists()
    assert store_path.exists()

    saved_train = pd.read_parquet(
        train_path
    )
    saved_store = pd.read_parquet(
        store_path
    )

    pd.testing.assert_frame_equal(
        saved_train,
        train,
    )
    pd.testing.assert_frame_equal(
        saved_store,
        store,
    )


def test_ingest_orchestrates_complete_lifecycle(
    monkeypatch,
):
    train_full = build_training_frame(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )
    store = build_store_frame()
    train_base = train_full.iloc[
        [0]
    ].copy()
    simulation = train_full.iloc[
        [1]
    ].copy()
    batch = build_training_frame(
        [
            "2026-01-03",
        ]
    )
    final_train = pd.concat(
        [
            train_base,
            batch,
        ],
        ignore_index=True,
    )

    load_mock = Mock(
        return_value=(
            train_full,
            store,
        )
    )
    split_mock = Mock(
        return_value=(
            train_base,
            simulation,
        )
    )
    persist_simulation_mock = Mock()
    collect_mock = Mock(
        return_value=[
            batch,
        ]
    )
    merge_mock = Mock(
        return_value=final_train
    )
    persist_validated_mock = Mock()

    monkeypatch.setattr(
        "src.data.raw.ingest.get_path",
        lambda name: {
            "raw_data": "data/raw",
            "validated_data": "data/validation",
        }[name],
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.load_base_datasets",
        load_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.create_simulation_split",
        split_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.persist_simulation_source_if_missing",
        persist_simulation_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.collect_incremental_batches",
        collect_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.merge_training_batches",
        merge_mock,
    )
    monkeypatch.setattr(
        "src.data.raw.ingest.persist_validated_datasets",
        persist_validated_mock,
    )

    with patch.dict(
        os.environ,
        {
            "APP_ENV": "dev",
        },
    ):
        ingest()

    load_mock.assert_called_once_with(
        "data/raw"
    )
    split_mock.assert_called_once_with(
        train_full
    )
    persist_simulation_mock.assert_called_once_with(
        simulation,
        raw_path="data/raw",
    )
    collect_mock.assert_called_once_with(
        raw_path="data/raw",
        environment="dev",
    )
    merge_mock.assert_called_once_with(
        train_base,
        [
            batch,
        ],
    )
    persist_validated_mock.assert_called_once_with(
        final_train,
        store,
        validated_path="data/validation",
    )