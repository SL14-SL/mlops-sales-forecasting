from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingDecision,
    decide_retraining,
)
from ..monitoring.retraining_state import (
    build_retraining_state_path,
    decision_was_processed,
    record_successful_retraining,
)
from ..monitoring.signal_collector import (
    collect_retraining_signals,
)
from ..pipeline.project_factory import (
    build_project_training_pipeline,
)
from .lifecycle_adapter import (
    PrefectTrainingLifecycleResult,
    run_prefect_model_lifecycle,
)


@dataclass(frozen=True)
class AutoRetrainingResult:
    """Result of one automated retraining decision cycle."""

    status: str
    decision_id: str
    reasons: tuple[str, ...]
    candidate_run_id: str | None = None
    champion_promoted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return {
            "status": self.status,
            "decision_id": self.decision_id,
            "reasons": list(self.reasons),
            "candidate_run_id": (self.candidate_run_id),
            "champion_promoted": (self.champion_promoted),
        }


def _monitoring_path(
    config: Mapping[str, Any],
) -> str:
    """Return the configured monitoring storage path."""
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a valid 'paths' section.")

    monitoring_path = paths.get("monitoring")

    if not isinstance(monitoring_path, str) or not monitoring_path.strip():
        raise ValueError("Config path 'monitoring' must be a non-empty string.")

    return monitoring_path


def _lifecycle_state_payload(
    lifecycle: PrefectTrainingLifecycleResult,
) -> dict[str, Any]:
    """Translate a template lifecycle result into state data."""
    candidate_run_id = lifecycle.pipeline.pipeline.training.run_id

    if not isinstance(candidate_run_id, str) or not candidate_run_id.strip():
        raise RuntimeError("Training lifecycle returned no candidate run ID.")

    return {
        "candidate_run_id": candidate_run_id,
        "final_refit_run_id": None,
        "champion_promoted": (lifecycle.serving_release is not None),
    }


def _decision_result(
    decision: RetrainingDecision,
    *,
    status: str,
) -> AutoRetrainingResult:
    """Build a result without executing training."""
    return AutoRetrainingResult(
        status=status,
        decision_id=decision.decision_id,
        reasons=decision.reasons,
    )


def run_auto_retraining(
    *,
    config: Mapping[str, Any],
) -> AutoRetrainingResult:
    """Evaluate signals and run at most one training lifecycle."""
    signals = collect_retraining_signals(config=config)
    decision = decide_retraining(signals)

    if decision.action is RetrainingAction.BLOCK:
        return _decision_result(
            decision,
            status="blocked",
        )

    if decision.action is RetrainingAction.SKIP:
        return _decision_result(
            decision,
            status="skipped",
        )

    state_path = build_retraining_state_path(_monitoring_path(config))

    if decision_was_processed(
        decision.decision_id,
        state_path=state_path,
    ):
        return AutoRetrainingResult(
            status="duplicate",
            decision_id=decision.decision_id,
            reasons=("Decision was already processed.",),
        )

    pipeline = build_project_training_pipeline(config)
    lifecycle = run_prefect_model_lifecycle(
        pipeline=pipeline,
        pipeline_run_id=(decision.decision_id),
        mlflow_tags={
            "retraining.decision_id": (decision.decision_id),
            "retraining.trigger_types": ",".join(decision.trigger_types),
        },
    )
    lifecycle_state = _lifecycle_state_payload(lifecycle)

    persisted_state = record_successful_retraining(
        decision=decision,
        training_result=lifecycle_state,
        state_path=state_path,
    )

    return AutoRetrainingResult(
        status="retrained",
        decision_id=decision.decision_id,
        reasons=decision.reasons,
        candidate_run_id=(persisted_state["candidate_run_id"]),
        champion_promoted=bool(persisted_state["champion_promoted"]),
    )
