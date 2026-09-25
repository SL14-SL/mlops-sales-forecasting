from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlops_sales_forecasting.configs.paths import (
    join_uri,
)
from mlops_sales_forecasting.data.contracts import (
    DatasetSplits,
)
from mlops_sales_forecasting.storage.filesystem import (
    ensure_dir,
)
from mlops_sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_NORMAL_VALIDATION_DAYS = 14
DEFAULT_DRIFT_VALIDATION_DAYS = 7


def _date_column(
    config: Mapping[str, Any],
) -> str:
    data_config = config.get(
        "data",
        {},
    )
    value = data_config.get("time_column")

    if not isinstance(value, str) or not value:
        raise ValueError("Config must define data.time_column.")

    return value


def _validation_days(
    config: Mapping[str, Any],
    *,
    is_drift_run: bool,
) -> int:
    training_config = config.get(
        "training",
        {},
    )

    key = "drift_validation_days" if is_drift_run else "normal_validation_days"
    default = DEFAULT_DRIFT_VALIDATION_DAYS if is_drift_run else DEFAULT_NORMAL_VALIDATION_DAYS
    value = training_config.get(
        key,
        default,
    )

    if not isinstance(value, int) or value < 1:
        raise ValueError(f"training.{key} must be a positive integer.")

    return value


class RossmannDatasetSplitter:
    """Create leakage-safe chronological forecasting splits."""

    def __init__(
        self,
        *,
        is_drift_run: bool = False,
    ) -> None:
        self.is_drift_run = is_drift_run

    def split(
        self,
        features: pd.DataFrame,
        config: Mapping[str, Any],
    ) -> DatasetSplits:
        date_column = _date_column(config)
        validation_days = _validation_days(
            config,
            is_drift_run=self.is_drift_run,
        )

        if features.empty:
            raise ValueError("Feature dataset is empty.")

        if date_column not in features.columns:
            raise ValueError(f"Feature dataset is missing the date column '{date_column}'.")

        frame = features.copy()
        frame[date_column] = pd.to_datetime(
            frame[date_column],
            errors="coerce",
        )

        if frame[date_column].isna().any():
            raise ValueError("Feature dataset contains invalid dates.")

        frame = frame.sort_values(date_column).reset_index(drop=True)

        maximum_date = frame[date_column].max()
        validation_offset = pd.to_timedelta(
            validation_days - 1,
            unit="D",
        )
        validation_start = maximum_date - validation_offset

        train = frame.loc[frame[date_column] < validation_start].copy()
        validation = frame.loc[frame[date_column] >= validation_start].copy()

        status = "DRIFT" if self.is_drift_run else "NORMAL"

        if train.empty:
            raise ValueError(f"{status} training split is empty.")

        if validation.empty:
            raise ValueError(f"{status} validation split is empty.")

        if train[date_column].max() >= validation[date_column].min():
            raise ValueError("Training and validation periods overlap.")

        logger.info(
            "[%s] Chronological split complete | train_rows=%s | validation_rows=%s",
            status,
            len(train),
            len(validation),
        )

        return DatasetSplits(
            train=train,
            validation=validation,
        )


def persist_dataset_splits(
    splits: DatasetSplits,
    *,
    splits_path: str,
) -> dict[str, str]:
    """Persist train and validation datasets as Parquet."""
    ensure_dir(splits_path)
    train_path = join_uri(
        splits_path,
        "train.parquet",
    )
    validation_path = join_uri(
        splits_path,
        "val.parquet",
    )

    splits.train.to_parquet(
        train_path,
        index=False,
    )
    splits.validation.to_parquet(
        validation_path,
        index=False,
    )

    return {
        "train": train_path,
        "validation": validation_path,
    }
