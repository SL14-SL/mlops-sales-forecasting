from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.loader import get_path


def new_data_available() -> bool:
    batch_dir = Path(get_path("raw_data")) / "new_batches"
    if not batch_dir.exists():
        return False
    return any(batch_dir.glob("*.csv"))


def drift_detected() -> bool:
    drift_file = Path(get_path("monitoring")) / "feature_drift_history.parquet"
    if not drift_file.exists():
        return False

    df = pd.read_parquet(drift_file)
    if df.empty:
        return False

    if "drift_detected" in df.columns:
        return bool(df["drift_detected"].fillna(False).any())

    return False


def should_retrain() -> bool:
    return new_data_available() or drift_detected()