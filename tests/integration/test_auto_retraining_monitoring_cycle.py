from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.orchestration import (
    retraining_service,
)


def operational_config(
    tmp_path: Path,
) -> dict:
    """Build isolated operational storage configuration."""
    return {
        "paths": {
            "raw_data": str(tmp_path / "raw"),
            "predictions": str(tmp_path / "predictions"),
            "monitoring": str(tmp_path / "monitoring"),
            "features": str(tmp_path / "features"),
            "models": str(tmp_path / "models"),
        },
        "monitoring": {
            "feature_drift": {
                "enabled": True,
                "numeric_features": [
                    "CompetitionDistance",
                ],
                "categorical_features": [
                    "Promo",
                    "Store",
                    "StoreType",
                    "Assortment",
                    "StateHoliday",
                ],
                "minimum_samples": 50,
                "p_value_threshold": 0.01,
                "statistic_threshold": 0.10,
                "lookback_days": 14,
            },
            "retraining": {
                "minimum_new_training_rows": 500,
                "cooldown_hours": 168,
                "scheduled_interval_hours": 168,
                "maximum_new_training_rows": 1_000_000,
                "drift": {
                    "lookback_days": 14,
                    "consecutive_windows": 2,
                },
                "performance": {
                    "rolling_window": "7D",
                    "minimum_samples": 500,
                    "consecutive_windows": 2,
                    "rmse_limit": 1375.0,
                    "mae_limit": 990.0,
                    "absolute_bias_limit": 900.0,
                },
            },
        },
    }


def test_empty_operational_cycle_skips_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the real monitoring and policy chain without training."""
    build_pipeline = MagicMock()
    execute_lifecycle = MagicMock()

    monkeypatch.setattr(
        retraining_service,
        "build_project_training_pipeline",
        build_pipeline,
    )
    monkeypatch.setattr(
        retraining_service,
        "run_prefect_model_lifecycle",
        execute_lifecycle,
    )

    result = retraining_service.run_auto_retraining(config=operational_config(tmp_path))

    assert result.status == "skipped"
    assert result.candidate_run_id is None
    assert result.champion_promoted is False
    assert result.reasons == ("Insufficient new training rows: 0/500.",)

    build_pipeline.assert_not_called()
    execute_lifecycle.assert_not_called()

    assert not (tmp_path / "monitoring" / "performance_rolling.parquet").exists()
    assert not (tmp_path / "monitoring" / "feature_drift_history.parquet").exists()
