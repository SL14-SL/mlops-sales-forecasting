from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingDecision,
)
from mlops_sales_forecasting.orchestration import (
    retraining_service,
)


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

    build_pipeline.assert_called_once_with(project_config)
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
