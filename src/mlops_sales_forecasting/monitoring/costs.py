from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow
import pandas as pd


@dataclass(frozen=True)
class TrainingCostSummary:
    """Estimated training activity and cost for one time window."""

    window_days: int
    run_count: int
    total_duration_seconds: float
    average_duration_seconds: float
    total_estimated_cost: float
    average_estimated_cost: float
    hourly_rate: float
    currency: str


def _require_mapping(
    config: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = config.get(name)

    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a valid '{name}' section.")

    return value


def _require_positive_number(
    config: Mapping[str, Any],
    name: str,
) -> float:
    value = config.get(name)

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Config value '{name}' must be positive.")

    return float(value)


def load_training_runs(
    config: Mapping[str, Any],
    *,
    max_results: int = 1000,
) -> pd.DataFrame:
    """Load recent training runs from the configured MLflow experiment."""
    tracking = _require_mapping(
        config,
        "tracking",
    )

    tracking_uri = tracking.get("mlflow_tracking_uri")
    experiment_name = tracking.get("experiment_name")

    if not isinstance(tracking_uri, str) or not tracking_uri:
        raise ValueError("Config must contain a non-empty 'tracking.mlflow_tracking_uri' value.")

    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("Config must contain a non-empty 'tracking.experiment_name' value.")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return pd.DataFrame()

    return mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id,
        ],
        max_results=max_results,
        order_by=[
            "start_time DESC",
        ],
    )


def _normalize_completed_runs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(
            {
                "start_time": pd.Series(
                    dtype="datetime64[ns, UTC]",
                ),
                "end_time": pd.Series(
                    dtype="datetime64[ns, UTC]",
                ),
                "duration_seconds": pd.Series(
                    dtype="float64",
                ),
            }
        )

    required_columns = {
        "start_time",
        "end_time",
    }
    missing_columns = required_columns - set(runs.columns)

    if missing_columns:
        raise KeyError(f"Training runs are missing columns: {sorted(missing_columns)}.")

    normalized = runs.copy()
    normalized["start_time"] = pd.to_datetime(
        normalized["start_time"],
        errors="coerce",
        utc=True,
    )
    normalized["end_time"] = pd.to_datetime(
        normalized["end_time"],
        errors="coerce",
        utc=True,
    )

    normalized = normalized.dropna(
        subset=[
            "start_time",
            "end_time",
        ]
    )

    if "status" in normalized.columns:
        normalized = normalized.loc[normalized["status"].eq("FINISHED")]

    normalized["duration_seconds"] = (
        normalized["end_time"] - normalized["start_time"]
    ).dt.total_seconds()

    return normalized.loc[normalized["duration_seconds"] >= 0].copy()


def summarize_training_costs(
    runs: pd.DataFrame,
    *,
    window_days: int,
    hourly_rate: float,
    currency: str,
    evaluation_time: datetime | None = None,
) -> TrainingCostSummary:
    """Estimate training costs from completed MLflow run durations."""
    if window_days <= 0:
        raise ValueError("Cost window must be positive.")

    if hourly_rate <= 0:
        raise ValueError("Training hourly rate must be positive.")

    if not currency.strip():
        raise ValueError("Cost currency must not be empty.")

    normalized = _normalize_completed_runs(runs)
    now = evaluation_time or datetime.now(UTC)

    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    cutoff = now - timedelta(days=window_days)

    recent = normalized.loc[normalized["start_time"] >= cutoff]

    run_count = len(recent)
    total_duration = float(recent["duration_seconds"].sum())
    average_duration = total_duration / run_count if run_count else 0.0
    total_cost = total_duration / 3600.0 * hourly_rate
    average_cost = total_cost / run_count if run_count else 0.0

    return TrainingCostSummary(
        window_days=window_days,
        run_count=run_count,
        total_duration_seconds=round(
            total_duration,
            3,
        ),
        average_duration_seconds=round(
            average_duration,
            3,
        ),
        total_estimated_cost=round(
            total_cost,
            6,
        ),
        average_estimated_cost=round(
            average_cost,
            6,
        ),
        hourly_rate=float(hourly_rate),
        currency=currency,
    )


def build_monthly_cost_scenarios(
    *,
    average_training_cost: float,
    drift_triggered_runs: int,
) -> dict[str, dict[str, float | int]]:
    """Project monthly costs for common retraining frequencies."""
    if average_training_cost < 0:
        raise ValueError("Average training cost must not be negative.")

    if drift_triggered_runs < 0:
        raise ValueError("Drift-triggered run count must not be negative.")

    frequencies = {
        "daily": 30,
        "weekly": 4,
        "drift_triggered": drift_triggered_runs,
    }

    return {
        name: {
            "runs_per_month": run_count,
            "estimated_monthly_cost": round(
                average_training_cost * run_count,
                6,
            ),
        }
        for name, run_count in frequencies.items()
    }


def build_training_cost_report(
    config: Mapping[str, Any],
    *,
    runs: pd.DataFrame | None = None,
    evaluation_time: datetime | None = None,
) -> dict[str, Any]:
    """Build observed and projected training-cost information."""
    costs = _require_mapping(
        config,
        "costs",
    )
    training = _require_mapping(
        costs,
        "training",
    )
    scenarios = _require_mapping(
        costs,
        "scenarios",
    )

    enabled = bool(
        training.get(
            "enabled",
            True,
        )
    )

    if not enabled:
        return {
            "enabled": False,
        }

    window_days = int(
        training.get(
            "window_days",
            30,
        )
    )
    hourly_rate = _require_positive_number(
        training,
        "estimated_hourly_rate",
    )
    currency = str(
        training.get(
            "currency",
            "EUR",
        )
    )
    drift_triggered_runs = int(
        scenarios.get(
            "drift_triggered_runs_per_month",
            8,
        )
    )

    training_runs = load_training_runs(config) if runs is None else runs
    summary = summarize_training_costs(
        training_runs,
        window_days=window_days,
        hourly_rate=hourly_rate,
        currency=currency,
        evaluation_time=evaluation_time,
    )

    return {
        "enabled": True,
        "summary": asdict(summary),
        "scenarios": (
            build_monthly_cost_scenarios(
                average_training_cost=(summary.average_estimated_cost),
                drift_triggered_runs=(drift_triggered_runs),
            )
        ),
    }
