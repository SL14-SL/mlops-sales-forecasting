from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..training.contracts import (
    EvaluationResult,
    TrainingResult,
)
from .aliases import (
    AliasAssignment,
    assign_challenger,
)
from .promotion import PromotionPolicy
from .promotion_config import load_promotion_policy
from .promotion_service import (
    PromotionOutcome,
    evaluate_and_promote_candidate,
)
from .registry import (
    ModelRegistrationResult,
    register_approved_model,
)


@dataclass(frozen=True)
class CandidateLifecycleResult:
    """Result of finalizing one evaluated model candidate."""

    registration: ModelRegistrationResult
    challenger_assignment: AliasAssignment | None
    promotion: PromotionOutcome | None


def finalize_model_candidate(
    *,
    training_result: TrainingResult,
    evaluation_result: EvaluationResult,
    promotion_policy: PromotionPolicy,
    config: Mapping[str, Any],
    artifact_path: str = "model",
    logged_model_uri: str | None = None,
) -> CandidateLifecycleResult:
    """Register, alias and possibly promote a model candidate."""

    registration = register_approved_model(
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=config,
        artifact_path=artifact_path,
        model_uri=logged_model_uri,
    )

    if not registration.registered:
        return CandidateLifecycleResult(
            registration=registration,
            challenger_assignment=None,
            promotion=None,
        )

    challenger_assignment = assign_challenger(
        registration=registration,
        config=config,
    )

    promotion = evaluate_and_promote_candidate(
        registration=registration,
        evaluation=evaluation_result,
        policy=promotion_policy,
        config=config,
    )

    return CandidateLifecycleResult(
        registration=registration,
        challenger_assignment=challenger_assignment,
        promotion=promotion,
    )

def finalize_configured_model_candidate(
    *,
    training_result: TrainingResult,
    evaluation_result: EvaluationResult,
    config: Mapping[str, Any],
    artifact_path: str = "model",
    logged_model_uri: str | None = None,
) -> CandidateLifecycleResult:
    """Finalize a candidate using the configured promotion policy."""

    promotion_policy = load_promotion_policy(
        config
    )

    return finalize_model_candidate(
        training_result=training_result,
        evaluation_result=evaluation_result,
        promotion_policy=promotion_policy,
        config=config,
        artifact_path=artifact_path,
        logged_model_uri=logged_model_uri,
    )