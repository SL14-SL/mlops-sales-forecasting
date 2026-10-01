from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingDecision,
)
from mlops_sales_forecasting.orchestration import (
    retraining_service,
)


@pytest.fixture(autouse=True)
def mock_monitoring_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    result = MagicMock()
    result.ground_truth_rows = 10
    result.inference_rows = 8
    result.performance_updated = True
    result.performance_rows = 2
    result.feature_drift_updated = True
    result.feature_drift_rows = 3
    result.performance_reason = "Performance history refreshed."

    refresh = MagicMock(return_value=result)

    monkeypatch.setattr(
        retraining_service,
        "refresh_monitoring_signals",
        refresh,
    )

    return refresh


def config() -> dict:
    return {
        "paths": {
            "monitoring": "data/monitoring",
        },
    }


def decision(
    action: RetrainingAction,
) -> RetrainingDecision:
    return RetrainingDecision(
        action=action,
        decision_id="retrain-test-123",
        reasons=("Policy result.",),
        trigger_types=(
            ("performance_degradation",) if action is RetrainingAction.TRAIN_CANDIDATE else ()
        ),
        evidence={
            "dataset_version": "dataset-v1",
            "batch_ids": ("gt-batch-001",),
        },
    )


@pytest.mark.parametrize(
    ("action", "expected_status"),
    [
        (
            RetrainingAction.BLOCK,
            "blocked",
        ),
        (
            RetrainingAction.SKIP,
            "skipped",
        ),
    ],
)
def test_non_training_decisions_return_immediately(
    monkeypatch: pytest.MonkeyPatch,
    action: RetrainingAction,
    expected_status: str,
) -> None:
    build_pipeline = MagicMock()
    execute_lifecycle = MagicMock()

    monkeypatch.setattr(
        retraining_service,
        "collect_retraining_signals",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        retraining_service,
        "decide_retraining",
        MagicMock(return_value=decision(action)),
    )
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

    result = retraining_service.run_auto_retraining(config=config())

    assert result.status == expected_status
    assert result.decision_id == ("retrain-test-123")
    assert result.candidate_run_id is None
    assert result.champion_promoted is False

    build_pipeline.assert_not_called()
    execute_lifecycle.assert_not_called()


def test_duplicate_decision_skips_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_pipeline = MagicMock()
    execute_lifecycle = MagicMock()

    monkeypatch.setattr(
        retraining_service,
        "collect_retraining_signals",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        retraining_service,
        "decide_retraining",
        MagicMock(return_value=decision(RetrainingAction.TRAIN_CANDIDATE)),
    )
    processed = MagicMock(return_value=True)
    monkeypatch.setattr(
        retraining_service,
        "decision_was_processed",
        processed,
    )
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

    result = retraining_service.run_auto_retraining(config=config())

    assert result.status == "duplicate"
    assert result.reasons == ("Decision was already processed.",)

    processed.assert_called_once_with(
        "retrain-test-123",
        state_path=("data/monitoring/retraining_state.json"),
    )
    build_pipeline.assert_not_called()
    execute_lifecycle.assert_not_called()


def test_authorized_decision_executes_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock()
    lifecycle = MagicMock()
    lifecycle.pipeline.pipeline.training.run_id = "mlflow-run-123"
    lifecycle.serving_release = MagicMock()

    monkeypatch.setattr(
        retraining_service,
        "collect_retraining_signals",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        retraining_service,
        "decide_retraining",
        MagicMock(return_value=decision(RetrainingAction.TRAIN_CANDIDATE)),
    )
    monkeypatch.setattr(
        retraining_service,
        "decision_was_processed",
        MagicMock(return_value=False),
    )

    build_pipeline = MagicMock(return_value=pipeline)
    execute_lifecycle = MagicMock(return_value=lifecycle)
    persist_state = MagicMock(
        return_value={
            "candidate_run_id": ("mlflow-run-123"),
            "champion_promoted": True,
        }
    )

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
    monkeypatch.setattr(
        retraining_service,
        "record_successful_retraining",
        persist_state,
    )

    project_config = config()

    result = retraining_service.run_auto_retraining(config=project_config)

    assert result.status == "retrained"
    assert result.candidate_run_id == ("mlflow-run-123")
    assert result.champion_promoted is True

    expected_config = {
        **project_config,
        "training": {
            "is_drift_run": True,
        },
    }

    build_pipeline.assert_called_once_with(expected_config)
    execute_lifecycle.assert_called_once_with(
        pipeline=pipeline,
        pipeline_run_id=("retrain-test-123"),
        mlflow_tags={
            "retraining.decision_id": ("retrain-test-123"),
            "retraining.trigger_types": ("performance_degradation"),
        },
    )
    persist_state.assert_called_once_with(
        decision=decision(RetrainingAction.TRAIN_CANDIDATE),
        training_result={
            "candidate_run_id": ("mlflow-run-123"),
            "final_refit_run_id": None,
            "champion_promoted": True,
        },
        state_path=("data/monitoring/retraining_state.json"),
        recorded_at_utc=None,
    )


def test_monitoring_is_refreshed_before_signal_collection(
    monkeypatch: pytest.MonkeyPatch,
    mock_monitoring_refresh: MagicMock,
) -> None:
    call_order: list[str] = []

    mock_monitoring_refresh.side_effect = lambda **_: (
        call_order.append("refresh")
        or MagicMock(
            ground_truth_rows=0,
            inference_rows=0,
            performance_updated=False,
            performance_rows=0,
            feature_drift_updated=False,
            feature_drift_rows=0,
            performance_reason="No data.",
        )
    )

    collect_signals = MagicMock(side_effect=lambda **_: call_order.append("collect") or MagicMock())

    monkeypatch.setattr(
        retraining_service,
        "collect_retraining_signals",
        collect_signals,
    )
    monkeypatch.setattr(
        retraining_service,
        "decide_retraining",
        MagicMock(return_value=decision(RetrainingAction.SKIP)),
    )

    result = retraining_service.run_auto_retraining(config=config())

    assert result.status == "skipped"
    assert call_order == [
        "refresh",
        "collect",
    ]
    mock_monitoring_refresh.assert_called_once_with(
        config=config(),
        observed_at=None,
    )


def test_result_is_serializable() -> None:
    result = retraining_service.AutoRetrainingResult(
        status="skipped",
        decision_id="retrain-test-123",
        reasons=("No trigger.",),
    )

    assert result.to_dict() == {
        "status": "skipped",
        "decision_id": ("retrain-test-123"),
        "reasons": ["No trigger."],
        "candidate_run_id": None,
        "champion_promoted": False,
    }


def test_monitoring_refresh_failure_stops_cycle(
    monkeypatch: pytest.MonkeyPatch,
    mock_monitoring_refresh: MagicMock,
) -> None:
    mock_monitoring_refresh.side_effect = OSError("Monitoring storage unavailable.")
    collect_signals = MagicMock()

    monkeypatch.setattr(
        retraining_service,
        "collect_retraining_signals",
        collect_signals,
    )

    with pytest.raises(
        OSError,
        match="Monitoring storage unavailable",
    ):
        retraining_service.run_auto_retraining(config=config())

    collect_signals.assert_not_called()


def test_logical_evaluation_time_is_propagated(
    monkeypatch: pytest.MonkeyPatch,
    mock_monitoring_refresh: MagicMock,
) -> None:
    evaluated_at = pd.Timestamp("2015-06-04T00:00:00Z")
    collect_signals = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(
        retraining_service,
        "collect_retraining_signals",
        collect_signals,
    )
    monkeypatch.setattr(
        retraining_service,
        "decide_retraining",
        MagicMock(return_value=decision(RetrainingAction.SKIP)),
    )

    result = retraining_service.run_auto_retraining(
        config=config(),
        evaluated_at=evaluated_at,
    )

    assert result.status == "skipped"
    mock_monitoring_refresh.assert_called_once_with(
        config=config(),
        observed_at=evaluated_at.to_pydatetime(),
    )
    collect_signals.assert_called_once_with(
        config=config(),
        evaluated_at=evaluated_at,
    )


def test_scheduled_refresh_uses_normal_training_config() -> None:
    scheduled_decision = RetrainingDecision(
        action=(RetrainingAction.TRAIN_CANDIDATE),
        decision_id="retrain-scheduled",
        reasons=("Scheduled refresh.",),
        trigger_types=("scheduled_refresh",),
        evidence={},
    )

    result = retraining_service._build_retraining_config(
        config(),
        scheduled_decision,
    )

    assert result["training"]["is_drift_run"] is False


def test_performance_trigger_uses_drift_training_config() -> None:
    result = retraining_service._build_retraining_config(
        config(),
        decision(RetrainingAction.TRAIN_CANDIDATE),
    )

    assert result["training"]["is_drift_run"] is True
