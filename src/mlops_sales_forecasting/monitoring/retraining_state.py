import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ..configs.paths import join_uri
from ..storage.filesystem import (
    file_exists,
    read_text,
    write_text,
)
from .retraining_policy import RetrainingDecision

RETRAINING_STATE_FILENAME = "retraining_state.json"


def build_retraining_state_path(
    monitoring_path: str,
) -> str:
    """Return the persistent retraining-state location."""
    if not isinstance(monitoring_path, str) or not monitoring_path.strip():
        raise ValueError("Monitoring path must be a non-empty string.")

    return join_uri(
        monitoring_path,
        RETRAINING_STATE_FILENAME,
    )


def load_retraining_state(
    state_path: str,
) -> dict[str, Any]:
    """Load retraining state or return an empty initial state."""
    if not state_path.strip():
        raise ValueError("Retraining state path must not be empty.")

    if not file_exists(state_path):
        return {}

    try:
        payload = json.loads(read_text(state_path))
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def decision_was_processed(
    decision_id: str,
    *,
    state_path: str,
) -> bool:
    """Return whether a decision completed successfully before."""
    if not decision_id.strip():
        raise ValueError("Retraining decision ID must not be empty.")

    state = load_retraining_state(state_path)

    return state.get("last_decision_id") == decision_id


def record_successful_retraining(
    *,
    decision: RetrainingDecision,
    training_result: Mapping[str, Any],
    state_path: str,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    """Persist state after a completed retraining lifecycle."""
    candidate_run_id = training_result.get("candidate_run_id")

    if not isinstance(candidate_run_id, str) or not candidate_run_id.strip():
        raise ValueError("Training result does not contain candidate_run_id.")

    timestamp = recorded_at_utc or datetime.now(UTC).isoformat()

    payload = {
        "schema_version": 1,
        "last_decision_id": (decision.decision_id),
        "last_retrained_at_utc": timestamp,
        "action": decision.action.value,
        "trigger_types": list(decision.trigger_types),
        "reasons": list(decision.reasons),
        "dataset_version": (decision.evidence.get("dataset_version")),
        "performance_window_end": (decision.evidence.get("performance_window_end")),
        "drift_window_end": (decision.evidence.get("drift_window_end")),
        "candidate_run_id": candidate_run_id,
        "final_refit_run_id": (training_result.get("final_refit_run_id")),
        "champion_promoted": bool(
            training_result.get(
                "champion_promoted",
                False,
            )
        ),
        "processed_batch_ids": list(
            decision.evidence.get(
                "batch_ids",
                (),
            )
        ),
    }

    write_text(
        state_path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
    )

    return payload
