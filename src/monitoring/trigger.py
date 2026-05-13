from __future__ import annotations

import fsspec
import pandas as pd

from src.configs.loader import file_exists, get_path


def _list_files(directory: str, pattern: str) -> list[str]:
    glob_pattern = f"{str(directory).rstrip('/')}/{pattern}"
    fs, fs_pattern = fsspec.core.url_to_fs(glob_pattern)
    return [str(path) for path in fs.glob(fs_pattern)]


def new_data_available() -> bool:
    batch_dir = f"{get_path('raw_data')}/new_batches"
    return len(_list_files(batch_dir, "*.csv")) > 0


def drift_detected() -> bool:
    drift_file = f"{get_path('monitoring')}/feature_drift_history.parquet"

    if not file_exists(drift_file):
        return False

    df = pd.read_parquet(drift_file)

    if df.empty:
        return False

    if "drift_detected" in df.columns:
        return bool(df["drift_detected"].fillna(False).any())

    return False


def should_retrain() -> bool:
    return new_data_available() or drift_detected()