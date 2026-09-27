import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from mlops_sales_forecasting.monitoring.summary import (
    build_monitoring_summary,
    summarize_drift,
    summarize_performance,
)


def test_summarize_performance_uses_latest_window() -> None:
    history = pd.DataFrame(
        {
            "window_start": [
                "2026-09-01",
                "2026-09-02",
            ],
            "window_end": [
                "2026-09-07",
                "2026-09-08",
            ],
            "rmse": [
                900.0,
                800.0,
            ],
            "mae": [
                700.0,
                600.0,
            ],
            "bias": [
                -100.0,
                -50.0,
            ],
            "n_samples": [
                500,
                600,
            ],
        }
    )

    result = summarize_performance(history)

    assert result["available"] is True
    assert result["rmse"] == 800.0
    assert result["mae"] == 600.0
    assert result["n_samples"] == 600
    assert result["window_end"] == ("2026-09-08T00:00:00+00:00")


def test_summarize_drift_uses_latest_batch() -> None:
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-09-01T00:00:00Z",
                "2026-09-02T00:00:00Z",
                "2026-09-02T00:00:00Z",
            ],
            "feature": [
                "old_feature",
                "Promo",
                "StoreType",
            ],
            "drift_detected": [
                True,
                True,
                False,
            ],
        }
    )

    result = summarize_drift(history)

    assert result == {
        "available": True,
        "timestamp": ("2026-09-02T00:00:00+00:00"),
        "checked_features": 2,
        "drifted_features": 1,
        "drifted_feature_names": [
            "Promo",
        ],
    }


def test_build_monitoring_summary(
    tmp_path: Path,
) -> None:
    monitoring_path = tmp_path / "monitoring"
    monitoring_path.mkdir()

    pd.DataFrame(
        {
            "window_start": [
                "2026-09-01",
            ],
            "window_end": [
                "2026-09-07",
            ],
            "rmse": [
                800.0,
            ],
            "mae": [
                600.0,
            ],
            "bias": [
                -50.0,
            ],
            "n_samples": [
                600,
            ],
        }
    ).to_parquet(
        monitoring_path / "performance_rolling.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "timestamp": [
                "2026-09-07T00:00:00Z",
            ],
            "feature": [
                "Promo",
            ],
            "drift_detected": [
                False,
            ],
        }
    ).to_parquet(
        monitoring_path / "feature_drift_history.parquet",
        index=False,
    )

    (monitoring_path / "retraining_state.json").write_text(
        json.dumps(
            {
                "last_decision_id": ("retrain-123"),
                "last_retrained_at_utc": ("2026-09-07T01:00:00Z"),
                "action": "train_candidate",
                "trigger_types": [
                    "performance_degradation",
                ],
                "candidate_run_id": "run-123",
                "champion_promoted": True,
            }
        ),
        encoding="utf-8",
    )

    manager = MagicMock()
    manager.ready = True
    manager.active_release_id = "release-7"
    manager.last_reload_error = None

    result = build_monitoring_summary(
        config={
            "paths": {
                "monitoring": str(monitoring_path),
            },
        },
        model_manager=manager,
    )

    assert result["serving"] == {
        "ready": True,
        "active_release_id": "release-7",
        "last_reload_error": None,
    }
    assert result["performance"]["rmse"] == 800.0
    assert result["feature_drift"]["drifted_features"] == 0
    assert result["retraining"]["last_decision_id"] == "retrain-123"
