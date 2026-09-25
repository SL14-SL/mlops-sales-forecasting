from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from ..training.contracts import EvaluationResult
from .aliases import AliasAssignment, promote_to_champion
from .mlflow import load_mlflow_tracking_settings
from .promotion import (
    PromotionDecision,
    PromotionPolicy,
    evaluate_promotion,
)
from .registry import ModelRegistrationResult


@dataclass(frozen=True)
class PromotionOutcome:
    """Result of evaluating and possibly promoting a model candidate."""

    decision: PromotionDecision
    previous_champion_version: str | None
    champion_assignment: AliasAssignment | None


def _matches_error_code(
    error: MlflowException,
    expected_code: int,
) -> bool:
    """Match numeric and string MLflow error codes."""

    expected_name = ErrorCode.Name(
        expected_code
    )

    return error.error_code in {
        expected_code,
        expected_name,
    }


def _is_missing_alias_error(
    error: MlflowException,
) -> bool:
    """Return whether MLflow reports a missing registry alias."""

    if _matches_error_code(
        error,
        RESOURCE_DOES_NOT_EXIST,
    ):
        return True

    message = str(error).lower()

    return (
        _matches_error_code(
            error,
            INVALID_PARAMETER_VALUE,
        )
        and "alias" in message
        and "not found" in message
    )


def _load_champion_metrics(
    client: MlflowClient,
    model_name: str,
) -> tuple[str | None, Mapping[str, float] | None]:
    """Load the current champion version and its recorded run metrics."""

    try:
        champion = client.get_model_version_by_alias(
            name=model_name,
            alias="champion",
        )
    except MlflowException as error:
        if _is_missing_alias_error(error):
            return None, None
        raise

    if champion.run_id is None:
        raise ValueError(
            "The current champion model version does not reference "
            "an MLflow run."
        )

    run = client.get_run(champion.run_id)

    return str(champion.version), dict(run.data.metrics)


def evaluate_and_promote_candidate(
    registration: ModelRegistrationResult,
    evaluation: EvaluationResult,
    policy: PromotionPolicy,
    config: Mapping[str, Any],
) -> PromotionOutcome:
    """Evaluate a registered candidate and promote it when permitted."""

    if not registration.registered:
        raise ValueError(
            "Only registered model candidates can be promoted."
        )

    if registration.model_version is None:
        raise ValueError(
            "The registered candidate does not have a model version."
        )

    if not evaluation.approved:
        raise ValueError(
            "Only candidates approved by the quality gate can be promoted."
        )

    settings = load_mlflow_tracking_settings(config)
    client = MlflowClient(tracking_uri=settings.tracking_uri)

    champion_version, champion_metrics = _load_champion_metrics(
        client=client,
        model_name=registration.model_name,
    )

    decision = evaluate_promotion(
        candidate_metrics=evaluation.metrics,
        champion_metrics=champion_metrics,
        policy=policy,
    )

    assignment = None

    if decision.promote:
        assignment = promote_to_champion(
            registration=registration,
            config=config,
        )

    return PromotionOutcome(
        decision=decision,
        previous_champion_version=champion_version,
        champion_assignment=assignment,
    )