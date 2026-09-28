from dataclasses import dataclass

import pandas as pd

from .ground_truth import (
    DriftApplication,
    DriftScenario,
    apply_drift_scenario,
)


@dataclass(frozen=True)
class SimulatedDailyBatch:
    """One deterministic daily batch from a simulation pool."""

    day: int
    date: pd.Timestamp
    data: pd.DataFrame
    drift: DriftApplication


def normalize_simulation_pool(
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize a forecasting simulation pool."""
    required_columns = {
        "Store",
        "Date",
        "Sales",
        "Promo",
    }
    missing_columns = required_columns - set(pool.columns)

    if missing_columns:
        raise KeyError(f"Simulation pool is missing columns: {sorted(missing_columns)}.")

    if pool.empty:
        raise ValueError("Simulation pool must not be empty.")

    normalized = pool.copy()
    normalized["Date"] = pd.to_datetime(
        normalized["Date"],
        errors="coerce",
    )

    if normalized["Date"].isna().any():
        raise ValueError("Simulation pool contains invalid dates.")

    normalized["Sales"] = pd.to_numeric(
        normalized["Sales"],
        errors="raise",
    )
    normalized["Promo"] = pd.to_numeric(
        normalized["Promo"],
        errors="raise",
    )

    return normalized.sort_values(
        [
            "Date",
            "Store",
        ]
    ).reset_index(drop=True)


def count_simulation_days(
    pool: pd.DataFrame,
) -> int:
    """Return the number of unique dates in a simulation pool."""
    normalized = normalize_simulation_pool(pool)

    return int(normalized["Date"].nunique())


def build_simulated_daily_batch(
    pool: pd.DataFrame,
    *,
    day: int,
    scenario: DriftScenario,
) -> SimulatedDailyBatch:
    """Build one deterministic drift-adjusted daily batch."""
    if day < 1:
        raise ValueError("Simulation day must be positive.")

    normalized = normalize_simulation_pool(pool)
    dates = normalized["Date"].drop_duplicates().sort_values().tolist()

    if day > len(dates):
        raise IndexError(f"Simulation day exceeds available pool: {day}/{len(dates)}.")

    selected_date = pd.Timestamp(dates[day - 1])
    batch = normalized.loc[normalized["Date"].eq(selected_date)].copy()

    adjusted, drift = apply_drift_scenario(
        batch,
        current_day=day,
        scenario=scenario,
    )

    return SimulatedDailyBatch(
        day=day,
        date=selected_date,
        data=adjusted.reset_index(drop=True),
        drift=drift,
    )
