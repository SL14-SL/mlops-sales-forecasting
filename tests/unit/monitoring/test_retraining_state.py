import json

import pytest

from mlops_sales_forecasting.monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingDecision,
)
from mlops_sales_forecasting.monitoring.retraining_state import (
    build_retraining_state_path,
    decision_was_processed,
    load_retraining_state,
    record_successful_retraining,
)


def decision() -> RetrainingDecision:
    return RetrainingDecision(
        action=RetrainingAction.TRAIN_CANDIDATE,
        decision_id="retrain-test-123",
        reasons=("Persistent degradation.",),
        trigger_types=("performance_degradation",),
        evidence={
            "dataset_version": "batch-v1",
            "batch_ids": (
                "gt-batch-001",
                "gt-batch-002",
            ),
            "performance_window_end": ("2026-08-13T00:00:00Z"),
            "drift_window_end": None,
        },
    )


def test_build_retraining_state_path() -> None:
    result = build_retraining_state_path("data/monitoring")

    assert result == ("data/monitoring/retraining_state.json")


def test_missing_state_returns_empty_dict(
    tmp_path,
) -> None:
    state_path = str(tmp_path / "retraining_state.json")

    assert load_retraining_state(state_path) == {}


def test_invalid_state_returns_empty_dict(
    tmp_path,
) -> None:
    state_path = tmp_path / "retraining_state.json"
    state_path.write_text(
        "invalid-json",
        encoding="utf-8",
    )

    assert load_retraining_state(str(state_path)) == {}


def test_decision_was_processed(
    tmp_path,
) -> None:
    state_path = tmp_path / "retraining_state.json"
    state_path.write_text(
        json.dumps({"last_decision_id": ("retrain-test-123")}),
        encoding="utf-8",
    )

    assert decision_was_processed(
        "retrain-test-123",
        state_path=str(state_path),
    )
    assert not decision_was_processed(
        "retrain-other",
        state_path=str(state_path),
    )


def test_successful_retraining_is_persisted(
    tmp_path,
) -> None:
    state_path = str(tmp_path / "nested" / "retraining_state.json")

    result = record_successful_retraining(
        decision=decision(),
        training_result={
            "candidate_run_id": "run-123",
            "final_refit_run_id": None,
            "champion_promoted": False,
        },
        state_path=state_path,
        recorded_at_utc=("2026-09-25T12:00:00+00:00"),
    )

    assert result["candidate_run_id"] == ("run-123")
    assert result["champion_promoted"] is False
    assert result["processed_batch_ids"] == [
        "gt-batch-001",
        "gt-batch-002",
    ]

    persisted = json.loads(
        (tmp_path / "nested" / "retraining_state.json").read_text(encoding="utf-8")
    )

    assert persisted["last_decision_id"] == ("retrain-test-123")
    assert persisted["last_retrained_at_utc"] == "2026-09-25T12:00:00+00:00"


def test_result_without_candidate_is_rejected(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="candidate_run_id",
    ):
        record_successful_retraining(
            decision=decision(),
            training_result={
                "champion_promoted": False,
            },
            state_path=str(tmp_path / "retraining_state.json"),
        )
