import os

import pandas as pd

from src.configs.loader import get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURES = get_path("features")
SPLITS = get_path("splits")

NORMAL_VALIDATION_DAYS = 14
DRIFT_VALIDATION_DAYS = 7


def split(is_drift_run: bool = False):
    """
    Create chronological training and validation splits.

    Normal training uses all available historical data and the latest
    14 days for validation.

    Drift retraining uses all available historical data before the latest
    7-day validation window. Recent observations receive higher weights
    during model training.
    """
    input_file = f"{FEATURES}/features.parquet"

    logger.info(
        "Loading features for splitting from: %s",
        input_file,
    )

    df = pd.read_parquet(input_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    maximum_date = df["Date"].max()
    minimum_date = df["Date"].min()

    if is_drift_run:
        status = "DRIFT"

        validation_start = maximum_date - pd.Timedelta(
            days=DRIFT_VALIDATION_DAYS - 1
        )

        training_end = validation_start - pd.Timedelta(days=1)
        training_start = minimum_date

        logger.warning(
            "Drift detected. Applying recency-weighted split."
        )

        train = df[
            (df["Date"] >= training_start)
            & (df["Date"] <= training_end)
        ].copy()

        val = df[
            (df["Date"] >= validation_start)
            & (df["Date"] <= maximum_date)
        ].copy()

    else:
        status = "NORMAL"

        validation_start = maximum_date - pd.Timedelta(
            days=NORMAL_VALIDATION_DAYS - 1
        )

        training_start = minimum_date
        training_end = validation_start - pd.Timedelta(days=1)

        logger.info(
            "Applying normal chronological split."
        )

        train = df[
            (df["Date"] >= training_start)
            & (df["Date"] <= training_end)
        ].copy()

        val = df[
            (df["Date"] >= validation_start)
            & (df["Date"] <= maximum_date)
        ].copy()

    if train.empty:
        raise ValueError(
            f"{status} training split is empty."
        )

    if val.empty:
        raise ValueError(
            f"{status} validation split is empty."
        )

    train_maximum_date = train["Date"].max()
    validation_minimum_date = val["Date"].min()

    if train_maximum_date >= validation_minimum_date:
        raise ValueError(
            "Training and validation periods overlap."
        )

    os.makedirs(SPLITS, exist_ok=True)

    train.to_parquet(
        f"{SPLITS}/train.parquet",
        index=False,
    )

    val.to_parquet(
        f"{SPLITS}/val.parquet",
        index=False,
    )

    train_days = train["Date"].nunique()
    validation_days = val["Date"].nunique()

    logger.info("[%s] Data split complete.", status)

    logger.info(
        "[%s] Train set: %s to %s | days=%s | rows=%s",
        status,
        train["Date"].min().date(),
        train["Date"].max().date(),
        train_days,
        len(train),
    )

    logger.info(
        "[%s] Validation set: %s to %s | days=%s | rows=%s",
        status,
        val["Date"].min().date(),
        val["Date"].max().date(),
        validation_days,
        len(val),
    )


if __name__ == "__main__":
    split(is_drift_run=False)