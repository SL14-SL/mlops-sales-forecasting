from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger

from flows.training_flow import (
    training_pipeline,
)
from src.monitoring.retraining_policy import (
    RetrainingAction,
)
from src.monitoring.retraining_state import (
    decision_was_processed,
    record_successful_retraining,
)
from src.monitoring.trigger import (
    evaluate_retraining,
)


@flow(name="Auto Retrain Decision Flow")
def auto_retrain_flow() -> dict[str, Any]:
    logger = get_run_logger()

    decision = evaluate_retraining()

    logger.info(
        "Retraining decision evaluated | "
        f"action={decision.action.value} | "
        f"decision_id={decision.decision_id} | "
        f"triggers={list(decision.trigger_types)} | "
        f"reasons={list(decision.reasons)}"
    )

    if decision.action == RetrainingAction.BLOCK:
        logger.error(
            "Retraining blocked by policy | "
            f"decision_id={decision.decision_id}"
        )

        return {
            "status": "blocked",
            "decision_id": decision.decision_id,
            "reasons": list(decision.reasons),
        }

    if decision.action == RetrainingAction.SKIP:
        logger.info(
            "Retraining skipped by policy | "
            f"decision_id={decision.decision_id}"
        )

        return {
            "status": "skipped",
            "decision_id": decision.decision_id,
            "reasons": list(decision.reasons),
        }

    if decision_was_processed(
        decision.decision_id
    ):
        logger.info(
            "Retraining decision was already processed. "
            "Skipping duplicate run | "
            f"decision_id={decision.decision_id}"
        )

        return {
            "status": "duplicate",
            "decision_id": decision.decision_id,
            "reasons": [
                "Decision was already processed."
            ],
        }

    logger.info(
        "Policy authorized a new Candidate run | "
        f"decision_id={decision.decision_id}"
    )

    # force_run bypasses the old training-flow skip gate.
    # Promotion remains governed separately by the promotion policy.
    training_result = training_pipeline(
        force_run=True
    )

    if not isinstance(training_result, dict):
        raise RuntimeError(
            "Training pipeline did not return "
            "a result dictionary."
        )

    state = record_successful_retraining(
        decision=decision,
        training_result=training_result,
    )

    logger.info(
        "Retraining completed and state persisted | "
        f"decision_id={decision.decision_id} | "
        f"candidate_run_id="
        f"{state['candidate_run_id']} | "
        f"champion_promoted="
        f"{state['champion_promoted']}"
    )

    return {
        "status": "retrained",
        "decision_id": decision.decision_id,
        "candidate_run_id": (
            state["candidate_run_id"]
        ),
        "champion_promoted": (
            state["champion_promoted"]
        ),
    }


if __name__ == "__main__":
    auto_retrain_flow()