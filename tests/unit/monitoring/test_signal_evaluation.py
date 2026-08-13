import pandas as pd

from src.monitoring.signal_evaluation import (
    evaluate_performance_degradation,
    evaluate_persistent_feature_drift,
)


def test_old_drift_is_ignored():
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-07-01T00:00:00Z",
            ],
            "feature": ["Customers"],
            "drift_detected": [True],
        }
    )

    result = evaluate_persistent_feature_drift(
        history,
        evaluated_at=pd.Timestamp(
            "2026-08-13T00:00:00Z"
        ),
        lookback_days=14,
        consecutive_windows=2,
    )

    assert result.triggered is False


def test_drift_must_persist_for_same_feature():
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "feature": [
                "Customers",
                "Customers",
            ],
            "drift_detected": [True, True],
        }
    )

    result = evaluate_persistent_feature_drift(
        history,
        evaluated_at=pd.Timestamp(
            "2026-08-13T01:00:00Z"
        ),
        lookback_days=14,
        consecutive_windows=2,
    )

    assert result.triggered is True
    assert "Customers" in result.reason


def test_different_features_are_not_persistent():
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "feature": ["Promo", "Customers"],
            "drift_detected": [True, True],
        }
    )

    result = evaluate_persistent_feature_drift(
        history,
        evaluated_at=pd.Timestamp(
            "2026-08-13T01:00:00Z"
        ),
        lookback_days=14,
        consecutive_windows=2,
    )

    assert result.triggered is False


def test_performance_degradation_must_persist():
    history = pd.DataFrame(
        {
            "window_end": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "rmse": [1400.0, 1420.0],
            "mae": [1000.0, 1010.0],
            "bias": [100.0, 120.0],
        }
    )

    result = evaluate_performance_degradation(
        history,
        consecutive_windows=2,
        rmse_limit=1375.0,
        mae_limit=990.0,
        absolute_bias_limit=900.0,
    )

    assert result.triggered is True


def test_single_bad_performance_window_is_insufficient():
    history = pd.DataFrame(
        {
            "window_end": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "rmse": [1200.0, 1420.0],
            "mae": [800.0, 1010.0],
            "bias": [100.0, 120.0],
        }
    )

    result = evaluate_performance_degradation(
        history,
        consecutive_windows=2,
        rmse_limit=1375.0,
        mae_limit=990.0,
        absolute_bias_limit=900.0,
    )

    assert result.triggered is False


def test_large_persistent_bias_triggers_degradation():
    history = pd.DataFrame(
        {
            "window_end": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "rmse": [1200.0, 1210.0],
            "mae": [800.0, 810.0],
            "bias": [-950.0, -970.0],
        }
    )

    result = evaluate_performance_degradation(
        history,
        consecutive_windows=2,
        rmse_limit=1375.0,
        mae_limit=990.0,
        absolute_bias_limit=900.0,
    )

    assert result.triggered is True