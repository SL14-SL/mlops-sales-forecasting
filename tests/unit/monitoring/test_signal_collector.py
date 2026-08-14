from unittest.mock import MagicMock

import pandas as pd

from src.monitoring import signal_collector


def settings() -> dict:
    return {
        "minimum_new_training_rows": 500,
        "cooldown_hours": 168,
        "maximum_new_training_rows": 1_000_000,
        "drift": {
            "lookback_days": 14,
            "consecutive_windows": 2,
        },
        "performance": {
            "consecutive_windows": 2,
            "rmse_limit": 1375.0,
            "mae_limit": 990.0,
            "absolute_bias_limit": 900.0,
        },
    }


def test_collects_normalized_retraining_signals(
    monkeypatch,
):
    drift_history = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "feature": [
                "Customers",
                "Customers",
            ],
            "drift_detected": [
                True,
                True,
            ],
        }
    )

    performance_history = pd.DataFrame(
        {
            "window_end": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "rmse": [
                1400.0,
                1420.0,
            ],
            "mae": [
                1000.0,
                1010.0,
            ],
            "bias": [
                100.0,
                120.0,
            ],
        }
    )

    monkeypatch.setattr(
        signal_collector,
        "get_retraining_settings",
        MagicMock(
            return_value=settings()
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "get_path",
        MagicMock(
            side_effect=lambda name: {
                "raw_data": "data/raw",
                "monitoring": (
                    "data/monitoring"
                ),
            }[name]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_list_files",
        MagicMock(
            return_value=[
                "data/raw/new_batches/"
                "ground_truth_001.csv",
            ]
        ),
    )
    mock_batch_reader = MagicMock(
        return_value=(
            800,
            "batch-test-001",
            ("gt-batch-001",),
            True,
            "Ground Truth is valid.",
        )
    )

    monkeypatch.setattr(
        signal_collector,
        "_read_ground_truth_batches",
        mock_batch_reader,
    )
    monkeypatch.setattr(
        signal_collector,
        "_read_parquet_if_available",
        MagicMock(
            side_effect=[
                drift_history,
                performance_history,
            ]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "load_retraining_state",
        MagicMock(return_value={}),
    )

    signals = (
        signal_collector
        .collect_retraining_signals(
            evaluated_at=pd.Timestamp(
                "2026-08-13T01:00:00Z"
            )
        )
    )

    mock_batch_reader.assert_called_once_with(
        [
            "data/raw/new_batches/"
            "ground_truth_001.csv",
        ],
        processed_batch_ids=set(),
    )

    assert signals.batch_ids == (
        "gt-batch-001",
    )

    assert signals.dataset_version == (
        "batch-test-001"
    )
    assert signals.new_training_rows == 800
    assert signals.data_quality_ok is True
    assert (
        signals.performance_degraded
        is True
    )
    assert (
        signals.feature_drift_persistent
        is True
    )
    assert signals.cooldown_active is False
    assert signals.budget_available is True


def test_recent_retraining_activates_cooldown():
    state = {
        "last_retrained_at_utc": (
            "2026-08-12T00:00:00Z"
        )
    }

    result = signal_collector._cooldown_active(
        state,
        evaluated_at=pd.Timestamp(
            "2026-08-13T00:00:00Z"
        ),
        cooldown_hours=168,
    )

    assert result is True


def test_expired_cooldown_is_inactive():
    state = {
        "last_retrained_at_utc": (
            "2026-08-01T00:00:00Z"
        )
    }

    result = signal_collector._cooldown_active(
        state,
        evaluated_at=pd.Timestamp(
            "2026-08-13T00:00:00Z"
        ),
        cooldown_hours=168,
    )

    assert result is False


def test_missing_retraining_state_has_no_cooldown(
    monkeypatch,
):
    monkeypatch.setattr(
        signal_collector,
        "load_retraining_state",
        MagicMock(return_value={}),
    )

    state = (
        signal_collector
        .load_retraining_state()
    )

    result = signal_collector._cooldown_active(
        state,
        evaluated_at=pd.Timestamp(
            "2026-08-13T00:00:00Z"
        ),
        cooldown_hours=168,
    )

    assert result is False


def test_rows_above_budget_are_reported(
    monkeypatch,
):
    empty_history = pd.DataFrame()

    monkeypatch.setattr(
        signal_collector,
        "get_retraining_settings",
        MagicMock(
            return_value=settings()
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "get_path",
        MagicMock(
            side_effect=lambda name: {
                "raw_data": "data/raw",
                "monitoring": (
                    "data/monitoring"
                ),
            }[name]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_list_files",
        MagicMock(
            return_value=[
                "data/raw/new_batches/"
                "ground_truth_001.csv",
            ]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_read_ground_truth_batches",
        MagicMock(
            return_value=(
                1_000_001,
                "batch-too-large",
                ("gt-batch-too-large",),
                True,
                "Ground Truth is valid.",
            )
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_read_parquet_if_available",
        MagicMock(
            side_effect=[
                empty_history,
                empty_history,
            ]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "load_retraining_state",
        MagicMock(return_value={}),
    )

    signals = (
        signal_collector
        .collect_retraining_signals(
            evaluated_at=pd.Timestamp(
                "2026-08-13T01:00:00Z"
            )
        )
    )

    assert signals.new_training_rows == (
        1_000_001
    )
    assert signals.budget_available is False

def test_processed_batch_rows_are_not_new(
    tmp_path,
):
    batch_path = (
        tmp_path
        / "ground_truth_test.csv"
    )
    batch_path.write_text(
        (
            "Store,Date,Sales,Customers,"
            "Open,Promo,StateHoliday,"
            "SchoolHoliday,DayOfWeek\n"
            "1,2015-04-28,1000,100,"
            "1,0,0,0,2\n"
        ),
        encoding="utf-8",
    )

    first_result = (
        signal_collector
        ._read_ground_truth_batches(
            [str(batch_path)],
            processed_batch_ids=set(),
        )
    )

    first_rows = first_result[0]
    batch_ids = first_result[2]

    assert first_result[3] is True
    assert batch_ids

    second_result = (
        signal_collector
        ._read_ground_truth_batches(
            [str(batch_path)],
            processed_batch_ids=set(
                batch_ids
            ),
        )
    )

    second_rows = second_result[0]
    second_batch_ids = second_result[2]

    assert first_rows == 1
    assert second_rows == 0
    assert second_batch_ids == batch_ids
    assert second_result[3] is True