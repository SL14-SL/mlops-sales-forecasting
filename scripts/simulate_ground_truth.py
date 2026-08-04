from datetime import datetime

import pandas as pd
import argparse

from src.configs.loader import get_path, join_uri, list_files, load_config, file_exists
from src.data.validation.validate import validate_train
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()

def apply_drift_scenario(
    batch_df: pd.DataFrame,
    *,
    current_day: int,
    scenario: str,
    drift_start_day: int,
    drift_duration_days: int,
    maximum_base_uplift: float,
    maximum_promo_uplift: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Apply a controlled demand regime shift.

    The default stable scenario leaves sales unchanged. The gradual promo
    scenario changes the relationship between promotions and sales over time.
    """
    result = batch_df.copy()

    if scenario == "stable":
        return result, {
            "progress": 0.0,
            "base_multiplier": 1.0,
            "promo_multiplier": 1.0,
        }

    if scenario != "gradual_promo_shift":
        raise ValueError(
            f"Unsupported drift scenario: {scenario}"
        )

    if current_day < drift_start_day:
        progress = 0.0
    else:
        elapsed_days = current_day - drift_start_day + 1
        progress = min(
            elapsed_days / drift_duration_days,
            1.0,
        )

    base_multiplier = (
        1.0
        + maximum_base_uplift * progress
    )
    promo_multiplier = (
        1.0
        + maximum_promo_uplift * progress
    )

    promo_mask = (
        pd.to_numeric(
            result["Promo"],
            errors="coerce",
        ).fillna(0)
        == 1
    )

    sales = pd.to_numeric(
        result["Sales"],
        errors="raise",
    ).astype(float)

    multipliers = pd.Series(
        base_multiplier,
        index=result.index,
        dtype=float,
    )

    multipliers.loc[
        promo_mask
    ] = promo_multiplier

    result["Sales"] = (
        sales
        .mul(multipliers)
        .round()
        .astype(int)
    )

    result["Sales"] = (
        result["Sales"]
        .round()
        .astype(int)
    )

    return result, {
        "progress": float(progress),
        "base_multiplier": float(base_multiplier),
        "promo_multiplier": float(promo_multiplier),
    }


def simulate_ground_truth_injection(
    *,
    scenario: str = "stable",
    drift_start_day: int = 46,
    drift_duration_days: int = 14,
    maximum_base_uplift: float = 0.10,
    maximum_promo_uplift: float = 0.35,
):
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


    unique_dates = sorted(df["Date"].unique())
    next_date = unique_dates[0]
    batch_data = df[df["Date"] == next_date].copy()

    batch_data, drift_metadata = apply_drift_scenario(
        batch_data,
        current_day=current_day_index,
        scenario=scenario,
        drift_start_day=drift_start_day,
        drift_duration_days=drift_duration_days,
        maximum_base_uplift=maximum_base_uplift,
        maximum_promo_uplift=maximum_promo_uplift,
    )

    logger.info(
        "Simulation regime | "
        "day=%s | scenario=%s | progress=%.3f | "
        "base_multiplier=%.3f | promo_multiplier=%.3f",
        current_day_index,
        scenario,
        drift_metadata["progress"],
        drift_metadata["base_multiplier"],
        drift_metadata["promo_multiplier"],
    )
    logger.info(f"Processing Day {current_day_index} for date: {next_date.date()}")


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
    parser = argparse.ArgumentParser(
        description=(
            "Inject one day of simulated forecasting ground truth."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "stable",
            "gradual_promo_shift",
        ],
        default="stable",
    )
    parser.add_argument(
        "--drift-start-day",
        type=int,
        default=46,
    )
    parser.add_argument(
        "--drift-duration-days",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--maximum-base-uplift",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--maximum-promo-uplift",
        type=float,
        default=0.35,
    )

    args = parser.parse_args()

    simulate_ground_truth_injection(
        scenario=args.scenario,
        drift_start_day=args.drift_start_day,
        drift_duration_days=args.drift_duration_days,
        maximum_base_uplift=args.maximum_base_uplift,
        maximum_promo_uplift=args.maximum_promo_uplift,
    )