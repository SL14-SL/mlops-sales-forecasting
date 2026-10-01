from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)

from mlops_sales_forecasting.tracking.promotion import (
    MetricDirection,
    PromotionPolicy,
)
from mlops_sales_forecasting.tracking.promotion_service import (
    evaluate_and_promote_candidate,
)

CONFIG = {
    "tracking": {
        "mlflow_tracking_uri": "http://localhost:5000",
        "experiment_name": "promotion-test",
        "model_name": "promotion-test-model",
    }
}

POLICY = PromotionPolicy(
    metric_name="score",
    direction=MetricDirection.MAXIMIZE,
    minimum_improvement=0.05,
)


def build_registration() -> MagicMock:
    registration = MagicMock()
    registration.registered = True
    registration.model_name = "promotion-test-model"
    registration.model_version = "7"
    return registration


def build_evaluation(score: float = 0.90) -> MagicMock:
    evaluation = MagicMock()
    evaluation.approved = True
    evaluation.metrics = {"score": score}
    return evaluation


def build_client_with_champion(
    champion_score: float,
) -> MagicMock:
    client = MagicMock()

    champion = MagicMock()
    champion.version = "3"
    champion.run_id = "champion-run"
    client.get_model_version_by_alias.return_value = champion

    run = MagicMock()
    run.data.metrics = {"score": champion_score}
    client.get_run.return_value = run

    return client


@patch("mlops_sales_forecasting.tracking.promotion_service.promote_to_champion")
@patch("mlops_sales_forecasting.tracking.promotion_service.MlflowClient")
def test_better_candidate_is_promoted(
    client_class: MagicMock,
    promote_to_champion: MagicMock,
) -> None:
    client_class.return_value = build_client_with_champion(0.80)
    assignment = MagicMock()
    promote_to_champion.return_value = assignment

    outcome = evaluate_and_promote_candidate(
        registration=build_registration(),
        evaluation=build_evaluation(0.90),
        policy=POLICY,
        config=CONFIG,
    )

    assert outcome.decision.promote is True
    assert outcome.previous_champion_version == "3"
    assert outcome.champion_assignment is assignment
    promote_to_champion.assert_called_once()


@patch("mlops_sales_forecasting.tracking.promotion_service.promote_to_champion")
@patch("mlops_sales_forecasting.tracking.promotion_service.MlflowClient")
def test_worse_candidate_is_not_promoted(
    client_class: MagicMock,
    promote_to_champion: MagicMock,
) -> None:
    client_class.return_value = build_client_with_champion(0.90)

    outcome = evaluate_and_promote_candidate(
        registration=build_registration(),
        evaluation=build_evaluation(0.91),
        policy=POLICY,
        config=CONFIG,
    )

    assert outcome.decision.promote is False
    assert outcome.previous_champion_version == "3"
    assert outcome.champion_assignment is None
    promote_to_champion.assert_not_called()


@patch("mlops_sales_forecasting.tracking.promotion_service.promote_to_champion")
@patch("mlops_sales_forecasting.tracking.promotion_service.MlflowClient")
def test_first_candidate_is_promoted(
    client_class: MagicMock,
    promote_to_champion: MagicMock,
) -> None:
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException(
        "Alias does not exist.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    client_class.return_value = client

    assignment = MagicMock()
    promote_to_champion.return_value = assignment

    outcome = evaluate_and_promote_candidate(
        registration=build_registration(),
        evaluation=build_evaluation(),
        policy=POLICY,
        config=CONFIG,
    )

    assert outcome.decision.promote is True
    assert outcome.previous_champion_version is None
    assert outcome.champion_assignment is assignment
    promote_to_champion.assert_called_once()


@patch("mlops_sales_forecasting.tracking.promotion_service.MlflowClient")
def test_unexpected_mlflow_error_is_propagated(
    client_class: MagicMock,
) -> None:
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException(
        "MLflow server unavailable.",
        error_code=INTERNAL_ERROR,
    )
    client_class.return_value = client

    with pytest.raises(
        MlflowException,
        match="MLflow server unavailable",
    ):
        evaluate_and_promote_candidate(
            registration=build_registration(),
            evaluation=build_evaluation(),
            policy=POLICY,
            config=CONFIG,
        )


def test_unregistered_candidate_is_rejected() -> None:
    registration = build_registration()
    registration.registered = False

    with pytest.raises(
        ValueError,
        match="Only registered model candidates",
    ):
        evaluate_and_promote_candidate(
            registration=registration,
            evaluation=build_evaluation(),
            policy=POLICY,
            config=CONFIG,
        )


def test_candidate_rejected_by_quality_gate_is_rejected() -> None:
    evaluation = build_evaluation()
    evaluation.approved = False

    with pytest.raises(
        ValueError,
        match="Only candidates approved",
    ):
        evaluate_and_promote_candidate(
            registration=build_registration(),
            evaluation=evaluation,
            policy=POLICY,
            config=CONFIG,
        )


@patch("mlops_sales_forecasting.tracking.promotion_service.promote_to_champion")
@patch("mlops_sales_forecasting.tracking.promotion_service.MlflowClient")
def test_first_candidate_handles_real_mlflow_alias_error(
    client_class: MagicMock,
    promote_to_champion: MagicMock,
) -> None:
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException(
        ("Registered model alias champion not found."),
        error_code=(INVALID_PARAMETER_VALUE),
    )
    client_class.return_value = client

    assignment = MagicMock()
    promote_to_champion.return_value = assignment

    outcome = evaluate_and_promote_candidate(
        registration=build_registration(),
        evaluation=build_evaluation(),
        policy=POLICY,
        config=CONFIG,
    )

    assert outcome.decision.promote is True
    assert outcome.previous_champion_version is None
    assert outcome.champion_assignment is assignment
    promote_to_champion.assert_called_once()


@patch("mlops_sales_forecasting.tracking.promotion_service.promote_to_champion")
@patch("mlops_sales_forecasting.tracking.promotion_service.MlflowClient")
def test_current_champion_metrics_provider_is_used(
    client_class: MagicMock,
    promote_to_champion: MagicMock,
) -> None:
    client = build_client_with_champion(0.60)
    client_class.return_value = client

    metrics_provider = MagicMock(
        return_value={
            "score": 0.80,
        }
    )
    assignment = MagicMock()
    promote_to_champion.return_value = assignment

    outcome = evaluate_and_promote_candidate(
        registration=build_registration(),
        evaluation=build_evaluation(0.90),
        policy=POLICY,
        config=CONFIG,
        champion_metrics_provider=(metrics_provider),
    )

    assert outcome.decision.promote is True
    assert outcome.decision.candidate_value == (0.90)
    assert outcome.decision.champion_value == (0.80)
    assert outcome.previous_champion_version == ("3")
    assert outcome.champion_assignment is assignment

    metrics_provider.assert_called_once_with("models:/promotion-test-model/3")
    client.get_run.assert_not_called()
    promote_to_champion.assert_called_once()


def test_failed_guardrail_prevents_promotion() -> None:
    client = build_client_with_champion(0.60)
    metrics_provider = MagicMock(
        return_value={
            "score": 0.80,
        }
    )
    guardrail = MagicMock()
    guardrail.return_value.approved = False
    guardrail.return_value.reasons = (
        "Candidate segment 'non_promo' RMSE regression exceeds the allowed limit.",
    )

    with (
        patch(
            "mlops_sales_forecasting.tracking.promotion_service.MlflowClient",
            return_value=client,
        ),
        patch(
            "mlops_sales_forecasting.tracking.promotion_service.promote_to_champion",
        ) as promote,
    ):
        outcome = evaluate_and_promote_candidate(
            registration=(build_registration()),
            evaluation=(build_evaluation(0.90)),
            policy=POLICY,
            config=CONFIG,
            champion_metrics_provider=(metrics_provider),
            promotion_guardrail=(guardrail),
        )

    assert outcome.decision.promote is False
    assert "failed paired promotion guardrails" in outcome.decision.reason
    assert "non_promo" in outcome.decision.reason
    assert outcome.champion_assignment is None

    guardrail.assert_called_once_with(
        candidate_metrics={
            "score": 0.90,
        },
        champion_metrics={
            "score": 0.80,
        },
    )
    promote.assert_not_called()
