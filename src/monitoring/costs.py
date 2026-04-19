from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import mlflow
import pandas as pd

from src.configs.loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
ENV_CFG = load_config()

drift_cfg = ENV_CFG.get("costs", {}).get("scenarios", {})
drift_runs = drift_cfg.get("drift_triggered_runs_per_month", 8)

@dataclass
class CostWindowSummary:
    window_days: int
    run_count: int
    total_training_cost: float
    total_training_duration_seconds: float
    avg_training_cost: float
    avg_training_duration_seconds: float
    currency: str


def get_tracking_uri() -> str | None:
    tracking_cfg = ENV_CFG.get("tracking", {})
    return tracking_cfg.get("mlflow_tracking_uri")


def get_experiment_name() -> str:
    return ENV_CFG.get("project_name", "ml-project")


def load_training_runs(max_results: int = 1000) -> pd.DataFrame:
    tracking_uri = get_tracking_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(get_experiment_name())
    if experiment is None:
        logger.warning("No MLflow experiment found.")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    if runs.empty:
        return runs

    runs["start_time"] = pd.to_datetime(runs["start_time"], utc=True, errors="coerce")
    return runs


def summarize_training_costs(runs: pd.DataFrame, window_days: int = 7) -> CostWindowSummary:
    if runs.empty:
        return CostWindowSummary(
            window_days=window_days,
            run_count=0,
            total_training_cost=0.0,
            total_training_duration_seconds=0.0,
            avg_training_cost=0.0,
            avg_training_duration_seconds=0.0,
            currency="EUR",
        )

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    recent = runs[runs["start_time"] >= cutoff].copy()

    cost_col = "metrics.estimated_training_cost"
    dur_col = "metrics.training_duration_seconds"
    currency_col = "params.cost_currency"

    if cost_col not in recent.columns:
        recent[cost_col] = 0.0
    if dur_col not in recent.columns:
        recent[dur_col] = 0.0
    if currency_col not in recent.columns:
        recent[currency_col] = "EUR"

    recent[cost_col] = pd.to_numeric(recent[cost_col], errors="coerce").fillna(0.0)
    recent[dur_col] = pd.to_numeric(recent[dur_col], errors="coerce").fillna(0.0)

    run_count = len(recent)
    total_cost = float(recent[cost_col].sum())
    total_duration = float(recent[dur_col].sum())
    avg_cost = total_cost / run_count if run_count else 0.0
    avg_duration = total_duration / run_count if run_count else 0.0
    currency = recent[currency_col].dropna().iloc[0] if run_count else "EUR"

    return CostWindowSummary(
        window_days=window_days,
        run_count=run_count,
        total_training_cost=round(total_cost, 6),
        total_training_duration_seconds=round(total_duration, 3),
        avg_training_cost=round(avg_cost, 6),
        avg_training_duration_seconds=round(avg_duration, 3),
        currency=currency,
    )


def build_monthly_cost_scenarios(avg_training_cost: float) -> dict[str, Any]:
    return {
        "daily_retraining": {
            "runs_per_month": 30,
            "estimated_monthly_cost": round(avg_training_cost * 30, 6),
        },
        "weekly_retraining": {
            "runs_per_month": 4,
            "estimated_monthly_cost": round(avg_training_cost * 4, 6),
        },
        "drift_triggered_retraining": {
            "runs_per_month": drift_runs,
            "estimated_monthly_cost": round(avg_training_cost * drift_runs, 6),
        },
    }


def build_cost_interpretation(summary: CostWindowSummary, scenarios: dict[str, Any]) -> str:
    if summary.run_count == 0:
        return "No recent training runs found, so no cost estimate is available yet."

    monthly_daily = scenarios["daily_retraining"]["estimated_monthly_cost"]

    if monthly_daily < 1:
        return (
            "Current training costs are very low. Daily retraining appears financially feasible "
            "at the current scale."
        )
    if monthly_daily < 20:
        return (
            "Current training costs are moderate. Daily retraining seems feasible, but weekly or "
            "event-based retraining may improve efficiency."
        )
    return (
        "Training costs are becoming material. Retraining frequency should be aligned with business "
        "impact and monitoring signals."
    )


def build_cost_report(window_days: int = 7) -> dict[str, Any]:
    runs = load_training_runs()
    summary = summarize_training_costs(runs, window_days=window_days)
    scenarios = build_monthly_cost_scenarios(summary.avg_training_cost)
    # interpretation = build_cost_interpretation(summary, scenarios)

    return {
        "summary": {
            "window_days": summary.window_days,
            "run_count": summary.run_count,
            "currency": summary.currency,
            "total_training_cost": summary.total_training_cost,
            "total_training_duration_seconds": summary.total_training_duration_seconds,
            "avg_training_cost": summary.avg_training_cost,
            "avg_training_duration_seconds": summary.avg_training_duration_seconds,
        },
        "scenarios": scenarios,
        # "interpretation": interpretation,
    }