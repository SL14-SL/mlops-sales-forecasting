from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import SimulationDayResult

_REQUIRED_COLUMNS = {
    "day",
    "cumulative_days",
    "rmse",
    "mae",
    "bias",
    "n_samples",
    "window_start",
    "window_end",
    "event",
    "champion_promoted",
    "scenario",
    "retraining_enabled",
    "drift_start_day",
    "drift_duration_days",
    "maximum_base_uplift",
    "maximum_promo_uplift",
}


def results_to_frame(
    results: Iterable[SimulationDayResult],
) -> pd.DataFrame:
    """Convert simulation results to a stable tabular schema."""
    records = [result.to_record() for result in results]

    if not records:
        return pd.DataFrame(
            columns=[
                "day",
                "latest_batch_file",
                "cumulative_days",
                "rmse",
                "mae",
                "bias",
                "n_samples",
                "window_start",
                "window_end",
                "event",
                "champion_promoted",
                "candidate_run_id",
                "scenario",
                "retraining_enabled",
                "drift_start_day",
                "drift_duration_days",
                "maximum_base_uplift",
                "maximum_promo_uplift",
            ]
        )

    frame = pd.DataFrame(records)

    return frame[
        [
            "day",
            "latest_batch_file",
            "cumulative_days",
            "rmse",
            "mae",
            "bias",
            "n_samples",
            "window_start",
            "window_end",
            "event",
            "champion_promoted",
            "candidate_run_id",
            "scenario",
            "retraining_enabled",
            "drift_start_day",
            "drift_duration_days",
            "maximum_base_uplift",
            "maximum_promo_uplift",
        ]
    ]


def write_lifecycle_results(
    results: Iterable[SimulationDayResult],
    path: str | Path,
) -> pd.DataFrame:
    """Write lifecycle simulation results as a CSV file."""
    frame = results_to_frame(results)
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame.to_csv(
        output_path,
        index=False,
    )

    return frame


def _parse_boolean_series(
    series: pd.Series,
) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()

    mapped = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )

    return mapped.fillna(False).astype(bool)


def normalize_lifecycle_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a lifecycle result loaded from CSV."""
    missing_columns = _REQUIRED_COLUMNS - set(frame.columns)

    if missing_columns:
        raise KeyError(f"Lifecycle result is missing columns: {sorted(missing_columns)}.")

    normalized = frame.copy()

    for column in (
        "day",
        "cumulative_days",
        "rmse",
        "mae",
        "bias",
        "n_samples",
        "drift_start_day",
        "drift_duration_days",
        "maximum_base_uplift",
        "maximum_promo_uplift",
    ):
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    for column in (
        "window_start",
        "window_end",
    ):
        normalized[column] = pd.to_datetime(
            normalized[column],
            errors="coerce",
        )

    normalized["champion_promoted"] = _parse_boolean_series(normalized["champion_promoted"])
    normalized["retraining_enabled"] = _parse_boolean_series(normalized["retraining_enabled"])

    normalized["event"] = normalized["event"].where(
        normalized["event"].notna(),
        None,
    )

    return (
        normalized.dropna(
            subset=[
                "day",
            ]
        )
        .sort_values("day")
        .reset_index(drop=True)
    )


def load_lifecycle_results(
    path: str | Path,
) -> pd.DataFrame:
    """Load and normalize one lifecycle result CSV."""
    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Lifecycle result not found: {input_path}")

    frame = pd.read_csv(input_path)

    return normalize_lifecycle_frame(frame)


def summarize_simulation_comparison(
    without_retraining: pd.DataFrame,
    with_retraining: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize final performance of two matching runs."""
    if without_retraining.empty or with_retraining.empty:
        raise ValueError("Both lifecycle runs must contain results.")

    without = normalize_lifecycle_frame(without_retraining)
    with_run = normalize_lifecycle_frame(with_retraining)

    final_without = without.iloc[-1]
    final_with = with_run.iloc[-1]
    rmse_without = float(final_without["rmse"])
    rmse_with = float(final_with["rmse"])

    improvement = (rmse_without - rmse_with) / rmse_without if rmse_without != 0 else 0.0

    retraining_events = int(with_run["event"].eq("retrain").sum())
    promotion_events = int(with_run["champion_promoted"].sum())

    return {
        "final_day": int(
            min(
                final_without["day"],
                final_with["day"],
            )
        ),
        "final_rmse_without_retraining": (rmse_without),
        "final_rmse_with_retraining": (rmse_with),
        "relative_rmse_improvement": float(improvement),
        "retraining_events": retraining_events,
        "promotion_events": promotion_events,
    }
