import os
import shutil

import pandas as pd

from src.configs.loader import get_path
from src.configs.paths import join_uri
from src.data.validation.validate import (
    validate_store,
    validate_train,
)
from src.storage.filesystem import (
    file_exists,
    list_files,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

SIMULATION_TRAINING_FRACTION = 0.9


def load_base_datasets(
    raw_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate the canonical training and store datasets.

    Args:
        raw_path: Local or GCS directory containing the raw source files.

    Returns:
        The validated training observations and store metadata.

    Raises:
        FileNotFoundError: If a required source file is unavailable.
        pandera.errors.SchemaError: If a dataset violates its schema.
    """
    train_path = join_uri(
        raw_path,
        "train.csv",
    )
    store_path = join_uri(
        raw_path,
        "store.csv",
    )

    logger.info(
        "Loading base datasets | "
        "train_path=%s | store_path=%s",
        train_path,
        store_path,
    )

    train = pd.read_csv(
        train_path,
        parse_dates=["Date"],
        dtype={
            "StateHoliday": str,
        },
    )
    store = pd.read_csv(
        store_path,
    )

    train = validate_train(
        train
    )
    store = validate_store(
        store
    )

    logger.info(
        "Base datasets loaded and validated | "
        "train_rows=%s | store_rows=%s",
        len(train),
        len(store),
    )

    return train, store


def create_simulation_split(
    train: pd.DataFrame,
    *,
    training_fraction: float = (
        SIMULATION_TRAINING_FRACTION
    ),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split observations chronologically into training and simulation data.

    The split is calculated over unique dates, ensuring that all observations
    from one date remain in the same partition.

    Args:
        train: Validated observations containing a Date column.
        training_fraction: Fraction of unique dates assigned to training.

    Returns:
        The historical training partition and future simulation partition.

    Raises:
        ValueError: If the fraction is invalid, dates cannot be parsed or
            either resulting partition would be empty.
    """
    if not 0 < training_fraction < 1:
        raise ValueError(
            "training_fraction must be "
            "between 0 and 1."
        )

    result = train.copy()
    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    if result["Date"].isna().any():
        raise ValueError(
            "Found invalid dates while creating "
            "the simulation split."
        )

    result = result.sort_values(
        "Date",
        ascending=True,
    )

    unique_dates = (
        result["Date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "At least two unique dates are required "
            "to create training and simulation partitions."
        )

    split_index = int(
        len(unique_dates)
        * training_fraction
    )

    split_index = min(
        max(split_index, 1),
        len(unique_dates) - 1,
    )

    split_date = pd.Timestamp(
        unique_dates[split_index]
    )

    train_base = result.loc[
        result["Date"] < split_date
    ].copy()

    simulation_truth = result.loc[
        result["Date"] >= split_date
    ].copy()

    if train_base.empty:
        raise ValueError(
            "Chronological training partition is empty."
        )

    if simulation_truth.empty:
        raise ValueError(
            "Chronological simulation partition is empty."
        )

    logger.info(
        "Chronological simulation split completed | "
        "split_date=%s | "
        "train_rows=%s | simulation_rows=%s | "
        "train_max_date=%s | simulation_min_date=%s",
        split_date.date(),
        len(train_base),
        len(simulation_truth),
        train_base["Date"].max().date(),
        simulation_truth["Date"].min().date(),
    )

    return (
        train_base,
        simulation_truth,
    )


def persist_simulation_source_if_missing(
    simulation_truth: pd.DataFrame,
    *,
    raw_path: str,
) -> None:
    """
    Persist the initial simulation Ground Truth when it does not yet exist.

    An existing simulation source is preserved to keep repeated demonstration
    runs reproducible.
    """
    simulation_path = join_uri(
        raw_path,
        "simulation_ground_truth.csv",
    )

    if file_exists(simulation_path):
        logger.info(
            "Existing simulation source preserved | "
            "path=%s",
            simulation_path,
        )
        return

    simulation_truth.to_csv(
        simulation_path,
        index=False,
    )

    logger.info(
        "Simulation source created | path=%s | rows=%s",
        simulation_path,
        len(simulation_truth),
    )


def load_and_validate_batch(
    batch_path: str,
) -> pd.DataFrame:
    """
    Load and validate one incremental training batch.

    Args:
        batch_path: Local or GCS path to a CSV batch.

    Returns:
        The validated batch.
    """
    batch = pd.read_csv(
        batch_path,
        parse_dates=["Date"],
        dtype={
            "StateHoliday": str,
        },
    )

    return validate_train(
        batch
    )


def collect_remote_batches(
    *,
    batch_directory: str,
) -> list[pd.DataFrame]:
    """
    Load valid incremental batches from remote storage.

    Invalid remote batches are logged and ignored. They remain in their source
    location because remote quarantine is not currently implemented.
    """
    batches: list[pd.DataFrame] = []

    batch_paths = list_files(
        join_uri(
            batch_directory,
            "*.csv",
        )
    )

    for batch_path in batch_paths:
        try:
            batch = load_and_validate_batch(
                batch_path
            )
        except Exception as error:
            logger.warning(
                "Remote batch rejected | "
                "path=%s | error=%s",
                batch_path,
                error,
            )
            continue

        batches.append(
            batch
        )

        logger.info(
            "Remote batch validated | path=%s | rows=%s",
            batch_path,
            len(batch),
        )

    return batches


def collect_local_batches(
    *,
    batch_directory: str,
    quarantine_directory: str,
) -> list[pd.DataFrame]:
    """
    Load valid local batches and move rejected files into quarantine.
    """
    if not os.path.isdir(
        batch_directory
    ):
        logger.info(
            "No local batch directory found | path=%s",
            batch_directory,
        )
        return []

    os.makedirs(
        quarantine_directory,
        exist_ok=True,
    )

    batches: list[pd.DataFrame] = []

    for file_name in sorted(
        os.listdir(batch_directory)
    ):
        if not file_name.endswith(
            ".csv"
        ):
            continue

        batch_path = join_uri(
            batch_directory,
            file_name,
        )

        try:
            batch = load_and_validate_batch(
                batch_path
            )

        except Exception as error:
            logger.warning(
                "Local batch rejected | "
                "path=%s | error=%s",
                batch_path,
                error,
            )

            quarantine_path = join_uri(
                quarantine_directory,
                file_name,
            )

            shutil.move(
                batch_path,
                quarantine_path,
            )

            logger.info(
                "Rejected batch moved to quarantine | "
                "source=%s | destination=%s",
                batch_path,
                quarantine_path,
            )
            continue

        batches.append(
            batch
        )

        logger.info(
            "Local batch validated | path=%s | rows=%s",
            batch_path,
            len(batch),
        )

    return batches


def collect_incremental_batches(
    *,
    raw_path: str,
    environment: str,
) -> list[pd.DataFrame]:
    """
    Collect and validate incremental training batches.

    Production and GCS environments use remote discovery. Local environments
    additionally move rejected batches into a quarantine directory.
    """
    batch_directory = join_uri(
        raw_path,
        "new_batches",
    )
    quarantine_directory = join_uri(
        raw_path,
        "quarantine",
    )

    if (
        environment == "prod"
        or raw_path.startswith("gs://")
    ):
        return collect_remote_batches(
            batch_directory=batch_directory,
        )

    return collect_local_batches(
        batch_directory=batch_directory,
        quarantine_directory=(
            quarantine_directory
        ),
    )


def merge_training_batches(
    train_base: pd.DataFrame,
    batches: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge and validate the historical training data and incremental batches.

    Returns:
        A chronologically sorted and validated training dataframe.
    """
    if not batches:
        logger.info(
            "No new training batches found."
        )
        return train_base.copy()

    merged = pd.concat(
        [
            train_base,
            *batches,
        ],
        ignore_index=True,
    )

    merged = merged.sort_values(
        "Date",
        ascending=True,
    )

    merged = validate_train(
        merged
    )

    logger.info(
        "Incremental batches integrated | "
        "batches=%s | total_rows=%s",
        len(batches),
        len(merged),
    )

    return merged


def persist_validated_datasets(
    train: pd.DataFrame,
    store: pd.DataFrame,
    *,
    validated_path: str,
) -> None:
    """
    Persist validated training observations and store metadata as Parquet.
    """
    if not validated_path.startswith(
        "gs://"
    ):
        os.makedirs(
            validated_path,
            exist_ok=True,
        )

    train_path = join_uri(
        validated_path,
        "train.parquet",
    )
    store_path = join_uri(
        validated_path,
        "store.parquet",
    )

    train.to_parquet(
        train_path,
        index=False,
    )
    store.to_parquet(
        store_path,
        index=False,
    )

    logger.info(
        "Validated datasets persisted | "
        "train_path=%s | store_path=%s | "
        "train_rows=%s | store_rows=%s",
        train_path,
        store_path,
        len(train),
        len(store),
    )


def ingest() -> None:
    """
    Execute the complete raw-data ingestion lifecycle.

    The lifecycle validates canonical source data, creates a fixed simulation
    partition, integrates valid incremental batches and persists canonical
    training artifacts for downstream feature generation.
    """
    raw_path = get_path(
        "raw_data"
    )
    validated_path = get_path(
        "validated_data"
    )
    environment = os.getenv(
        "APP_ENV",
        "dev",
    )

    logger.info(
        "Starting ingestion | source=%s | environment=%s",
        raw_path,
        environment,
    )

    train_full, store = (
        load_base_datasets(
            raw_path
        )
    )

    train_base, simulation_truth = (
        create_simulation_split(
            train_full
        )
    )

    persist_simulation_source_if_missing(
        simulation_truth,
        raw_path=raw_path,
    )

    batches = collect_incremental_batches(
        raw_path=raw_path,
        environment=environment,
    )

    final_train = merge_training_batches(
        train_base,
        batches,
    )

    persist_validated_datasets(
        final_train,
        store,
        validated_path=validated_path,
    )

    logger.info(
        "Ingestion completed successfully | "
        "total_rows=%s | output=%s",
        len(final_train),
        validated_path,
    )


if __name__ == "__main__":
    ingest()