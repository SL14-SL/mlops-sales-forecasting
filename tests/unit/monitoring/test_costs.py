from datetime import UTC, datetime

import pandas as pd
import pytest

from mlops_sales_forecasting.monitoring.costs import (
    build_monthly_cost_scenarios,
    build_training_cost_report,
    summarize_training_costs,
)


def build_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": pd.to_datetime(
                [
                    "2026-09-26T10:00:00Z",
                    "2026-09-25T10:00:00Z",
                    "2026-08-01T10:00:00Z",
                    "2026-09-26T12:00:00Z",
                ],
                utc=True,
            ),
            "end_time": pd.to_datetime(
                [
                    "2026-09-26T11:00:00Z",
                    "2026-09-25T10:30:00Z",
                    "2026-08-01T11:00:00Z",
                    "2026-09-26T13:00:00Z",
                ],
                utc=True,
            ),
            "status": [
                "FINISHED",
                "FINISHED",
                "FINISHED",
                "FAILED",
            ],
        }
    )


def test_summarize_training_costs() -> None:
    summary = summarize_training_costs(
        build_runs(),
        window_days=7,
        hourly_rate=0.40,
        currency="EUR",
        evaluation_time=datetime(
            2026,
            9,
            27,
            tzinfo=UTC,
        ),
    )

    assert summary.run_count == 2
    assert summary.total_duration_seconds == 5400.0
    assert summary.average_duration_seconds == 2700.0
    assert summary.total_estimated_cost == 0.6
    assert summary.average_estimated_cost == 0.3


def test_empty_runs_produce_zero_cost() -> None:
    summary = summarize_training_costs(
        pd.DataFrame(),
        window_days=30,
        hourly_rate=0.40,
        currency="EUR",
    )

    assert summary.run_count == 0
    assert summary.total_estimated_cost == 0.0
    assert summary.average_estimated_cost == 0.0


def test_monthly_cost_scenarios() -> None:
    result = build_monthly_cost_scenarios(
        average_training_cost=0.5,
        drift_triggered_runs=8,
    )

    assert result["daily"] == {
        "runs_per_month": 30,
        "estimated_monthly_cost": 15.0,
    }
    assert result["weekly"] == {
        "runs_per_month": 4,
        "estimated_monthly_cost": 2.0,
    }
    assert result["drift_triggered"] == {
        "runs_per_month": 8,
        "estimated_monthly_cost": 4.0,
    }


def test_build_training_cost_report() -> None:
    config = {
        "costs": {
            "training": {
                "enabled": True,
                "currency": "EUR",
                "estimated_hourly_rate": 0.40,
                "window_days": 7,
            },
            "scenarios": {
                "drift_triggered_runs_per_month": 8,
            },
        },
    }

    report = build_training_cost_report(
        config,
        runs=build_runs(),
        evaluation_time=datetime(
            2026,
            9,
            27,
            tzinfo=UTC,
        ),
    )

    assert report["enabled"] is True
    assert report["summary"]["run_count"] == 2
    assert report["scenarios"]["weekly"]["estimated_monthly_cost"] == 1.2


def test_disabled_cost_monitoring() -> None:
    report = build_training_cost_report(
        {
            "costs": {
                "training": {
                    "enabled": False,
                },
                "scenarios": {},
            },
        },
        runs=pd.DataFrame(),
    )

    assert report == {
        "enabled": False,
    }


@pytest.mark.parametrize(
    ("window_days", "hourly_rate"),
    [
        (0, 0.40),
        (7, 0.0),
    ],
)
def test_invalid_cost_settings_are_rejected(
    window_days: int,
    hourly_rate: float,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        summarize_training_costs(
            pd.DataFrame(),
            window_days=window_days,
            hourly_rate=hourly_rate,
            currency="EUR",
        )
