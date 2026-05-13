from datetime import datetime

import pandas as pd

from src.configs.loader import get_path, join_uri, list_files, load_config, file_exists
from src.data.validation.validate import validate_train
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()


def simulate_ground_truth_injection(drift_factor=1):
    """
    Simulates daily data injection by moving one day from the simulation pool
    to the active 'new_batches' directory. Applies drift after Day 5.

    Works for both local paths and gs:// paths.
    """
    raw_path = get_path("raw_data")
    source_path = join_uri(raw_path, "simulation_ground_truth.csv")
    target_dir = join_uri(raw_path, "new_batches")

    if not file_exists(source_path):
        raise FileNotFoundError(
            f"Simulation source not found at {source_path}. Run ingest.py first."
        )

    try:
        df = pd.read_csv(source_path, parse_dates=["Date"], dtype={"StateHoliday": str})
    except Exception as e:
        logger.error(f"Failed to read simulation pool: {e}")
        return

    if df.empty:
        logger.info("Simulation pool is empty.")
        print("Remaining days in pool: 0")
        return

    existing_batches = list_files(join_uri(target_dir, "ground_truth_*.csv"))
    current_day_index = len(existing_batches) + 1

    active_drift = 1.0 if current_day_index <= 5 else drift_factor

    if active_drift == 1.0:
        logger.info(f"Day {current_day_index}: Simulating STABLE data (No drift).")
    else:
        logger.warning(
            f"Day {current_day_index}: Simulating DRIFT (Factor: {active_drift})."
        )

    unique_dates = sorted(df["Date"].unique())
    next_date = unique_dates[0]
    batch_data = df[df["Date"] == next_date].copy()

    logger.info(f"Processing Day {current_day_index} for date: {next_date.date()}")

    batch_data["Sales"] = (batch_data["Sales"] * active_drift).astype(int)

    try:
        validate_train(batch_data)
    except Exception as e:
        logger.error(f"Validation failed for batch: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_filename = f"ground_truth_{timestamp}.csv"
    batch_path = join_uri(target_dir, batch_filename)

    batch_data.to_csv(batch_path, index=False)

    if batch_path.startswith("gs://"):
        logger.info(f"Cloud: Uploaded batch to GCS: {batch_filename}")
    else:
        logger.info(f"Local: Saved batch: {batch_filename}")

    remaining_pool = df[df["Date"] > next_date]
    remaining_pool.to_csv(source_path, index=False)

    num_remaining = remaining_pool["Date"].nunique()
    logger.info(f"Remaining days in pool: {num_remaining}")

    print(f"Remaining days in pool: {num_remaining}")


if __name__ == "__main__":
    simulate_ground_truth_injection()