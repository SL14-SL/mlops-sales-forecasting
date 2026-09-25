import pandas as pd
import pytest

from mlops_sales_forecasting.data.splits.split import (
    RossmannDatasetSplitter,
    persist_dataset_splits,
)


def config() -> dict:
    return {
        "data": {
            "time_column": "Date",
        },
        "training": {
            "normal_validation_days": 14,
            "drift_validation_days": 7,
        },
    }


def feature_frame(
    days: int = 30,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1 for _ in range(days)],
            "Date": pd.date_range(
                "2026-01-01",
                periods=days,
                freq="D",
            ),
            "Sales": [float(value) for value in range(days)],
        }
    )


def test_normal_split_uses_latest_fourteen_days() -> None:
    result = RossmannDatasetSplitter().split(
        feature_frame(),
        config(),
    )

    assert result.train["Date"].nunique() == 16
    assert result.validation["Date"].nunique() == 14
    assert result.train["Date"].max() < result.validation["Date"].min()


def test_drift_split_uses_latest_seven_days() -> None:
    result = RossmannDatasetSplitter(is_drift_run=True).split(
        feature_frame(),
        config(),
    )

    assert result.train["Date"].nunique() == 23
    assert result.validation["Date"].nunique() == 7


def test_split_rejects_insufficient_history() -> None:
    with pytest.raises(
        ValueError,
        match="training split is empty",
    ):
        RossmannDatasetSplitter().split(
            feature_frame(days=10),
            config(),
        )


def test_persist_dataset_splits(
    tmp_path,
) -> None:
    splits = RossmannDatasetSplitter().split(
        feature_frame(),
        config(),
    )

    paths = persist_dataset_splits(
        splits,
        splits_path=str(tmp_path),
    )

    assert (tmp_path / "train.parquet").is_file()
    assert (tmp_path / "val.parquet").is_file()
    assert paths == {
        "train": str(tmp_path / "train.parquet"),
        "validation": str(tmp_path / "val.parquet"),
    }
