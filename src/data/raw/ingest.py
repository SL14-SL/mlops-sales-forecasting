import os
import shutil

import pandas as pd

from src.configs.loader import file_exists, get_path, join_uri, list_files
from src.data.validation.validate import validate_store, validate_train
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingest():
    """
    Main ingestion task:
    - Loads raw data and validates it.
    - Performs a chronological 90/10 split for simulation.
    - Merges new batches with individual validation and quarantine logic.

    Works for both local paths and gs:// paths.
    """
    raw_path = get_path("raw_data")
    validated_path = get_path("validated_data")

    env = os.getenv("APP_ENV", "dev")
    logger.info(f"Starting ingestion process. Source: {raw_path} | Env: {env}")

    # 1. Load original Kaggle source files
    try:
        train_full = pd.read_csv(
            join_uri(raw_path, "train.csv"),
            parse_dates=["Date"],
            dtype={"StateHoliday": str},
        )
        store = pd.read_csv(join_uri(raw_path, "store.csv"))
        logger.info("Base source files loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load base source files: {e}")
        return

    # 2. Initial Validation
    validate_store(store)
    validate_train(train_full)
    logger.info("Initial validation of base data passed.")

    # 3. Chronological 90/10 Split on unique dates
    train_full = train_full.sort_values("Date", ascending=True).copy()
    train_full["Date"] = pd.to_datetime(train_full["Date"], errors="coerce")

    if train_full["Date"].isna().any():
        raise ValueError("Found invalid dates in training data during chronological split.")

    unique_dates = sorted(train_full["Date"].dropna().unique())
    date_split_idx = int(len(unique_dates) * 0.9)
    date_split_idx = min(max(date_split_idx, 0), len(unique_dates) - 1)

    split_date = pd.Timestamp(unique_dates[date_split_idx])

    train_base = train_full[train_full["Date"] < split_date].copy()
    sim_truth = train_full[train_full["Date"] >= split_date].copy()

    logger.info(
        "Chronological date-based split completed | "
        f"split_date={split_date.date()} | "
        f"train_rows={len(train_base)} | sim_rows={len(sim_truth)} | "
        f"train_max_date={train_base['Date'].max().date() if not train_base.empty else None} | "
        f"sim_min_date={sim_truth['Date'].min().date() if not sim_truth.empty else None}"
    )

    # Save simulation source if missing
    sim_source_path = join_uri(raw_path, "simulation_ground_truth.csv")

    if not file_exists(sim_source_path):
        sim_truth.to_csv(sim_source_path, index=False)
        logger.info(f"Created simulation source: {sim_source_path}")

    # 4. Collect incremental batches with quarantine logic
    final_train = train_base
    new_batches_found = []

    batch_dir = join_uri(raw_path, "new_batches")
    quarantine_dir = join_uri(raw_path, "quarantine")

    if env == "prod" or raw_path.startswith("gs://"):
        batch_files = list_files(join_uri(batch_dir, "*.csv"))

        for batch_file in batch_files:
            try:
                batch_df = pd.read_csv(
                    batch_file,
                    parse_dates=["Date"],
                    dtype={"StateHoliday": str},
                )
                validate_train(batch_df)
                new_batches_found.append(batch_df)
                logger.info(f"Cloud batch '{batch_file}' validated successfully.")

            except Exception as e:
                logger.warning(f"Cloud batch '{batch_file}' failed validation: {e}")

    else:
        if os.path.exists(batch_dir):
            os.makedirs(quarantine_dir, exist_ok=True)

            for file_name in os.listdir(batch_dir):
                if not file_name.endswith(".csv"):
                    continue

                batch_path = join_uri(batch_dir, file_name)

                try:
                    batch_df = pd.read_csv(
                        batch_path,
                        parse_dates=["Date"],
                        dtype={"StateHoliday": str},
                    )
                    validate_train(batch_df)
                    new_batches_found.append(batch_df)
                    logger.info(f"Batch '{file_name}' validated successfully.")

                except Exception as e:
                    logger.warning(f"Batch '{file_name}' rejected: {e}")
                    dest_path = join_uri(quarantine_dir, file_name)
                    shutil.move(batch_path, dest_path)
                    logger.info(f"Moved corrupted file '{file_name}' to quarantine.")

    # 5. Final merge and re-validation
    if new_batches_found:
        final_train = pd.concat([train_base] + new_batches_found, ignore_index=True)
        final_train = final_train.sort_values("Date", ascending=True)
        validate_train(final_train)
        logger.info(f"Integrated {len(new_batches_found)} new batches into training set.")

    # 6. Export to Parquet
    if not validated_path.startswith("gs://"):
        os.makedirs(validated_path, exist_ok=True)

    final_train.to_parquet(join_uri(validated_path, "train.parquet"), index=False)
    store.to_parquet(join_uri(validated_path, "store.parquet"), index=False)

    logger.info(
        f"Ingestion complete. Total rows: {len(final_train)} | Output: {validated_path}"
    )


if __name__ == "__main__":
    ingest()