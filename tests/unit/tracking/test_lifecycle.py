from unittest.mock import MagicMock, patch

from mlops_sales_forecasting.tracking.aliases import (
    AliasAssignment,
    ModelAlias,
)
from mlops_sales_forecasting.tracking.lifecycle import (
    finalize_configured_model_candidate,
    finalize_model_candidate,
)
from mlops_sales_forecasting.tracking.promotion import (
    MetricDirection,
    PromotionPolicy,
)
from mlops_sales_forecasting.tracking.registry import (
    ModelRegistrationResult,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)

CONFIG = {
    "tracking": {
        "mlflow_tracking_uri": "http://localhost:5000",
        "experiment_name": "lifecycle-test",
        "model_name": "lifecycle-test-model",
    }
}

POLICY = PromotionPolicy(
    metric_name="score",
    direction=MetricDirection.MAXIMIZE,
    minimum_improvement=0.05,
)


def build_training_result() -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id="run-1",
        metrics={"score": 0.90},
        parameters={"depth": 4},
        artifacts={"model": "models/model.pkl"},
    )


def build_approved_evaluation() -> EvaluationResult:
    return EvaluationResult(
        metrics={"score": 0.90},
        approved=True,
    )


def build_registration() -> ModelRegistrationResult:
    return ModelRegistrationResult(
        registered=True,
        run_id="run-1",
        model_name="lifecycle-test-model",
        model_version="7",
        model_uri="models:/lifecycle-test-model/7",
    )


@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "evaluate_and_promote_candidate"
)
@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "assign_challenger"
)
@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "register_approved_model"
)
def test_approved_candidate_completes_lifecycle(
    register_approved_model: MagicMock,
    assign_challenger: MagicMock,
    evaluate_and_promote_candidate: MagicMock,
) -> None:
    training_result = build_training_result()
    evaluation_result = build_approved_evaluation()
    registration = build_registration()

    challenger_assignment = AliasAssignment(
        model_name="lifecycle-test-model",
        model_version="7",
        alias=ModelAlias.CHALLENGER,
    )
    promotion = MagicMock()

    register_approved_model.return_value = registration
    assign_challenger.return_value = challenger_assignment
    evaluate_and_promote_candidate.return_value = promotion

    result = finalize_model_candidate(
        training_result=training_result,
        evaluation_result=evaluation_result,
        promotion_policy=POLICY,
        config=CONFIG,
    )

    assert result.registration is registration
    assert (
        result.challenger_assignment
        is challenger_assignment
    )
    assert result.promotion is promotion

    register_approved_model.assert_called_once_with(
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=CONFIG,
        artifact_path="model",
        model_uri=None,
    )
    assign_challenger.assert_called_once_with(
        registration=registration,
        config=CONFIG,
    )
    evaluate_and_promote_candidate.assert_called_once_with(
        registration=registration,
        evaluation=evaluation_result,
        policy=POLICY,
        config=CONFIG,
    )


@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "evaluate_and_promote_candidate"
)
@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "assign_challenger"
)
@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "register_approved_model"
)
def test_rejected_candidate_stops_after_registration_decision(
    register_approved_model: MagicMock,
    assign_challenger: MagicMock,
    evaluate_and_promote_candidate: MagicMock,
) -> None:
    training_result = build_training_result()
    evaluation_result = EvaluationResult(
        metrics={"score": 0.60},
        approved=False,
        reasons=("Score is below the quality gate.",),
    )
    registration = ModelRegistrationResult(
        registered=False,
        run_id="run-1",
        model_name="lifecycle-test-model",
        reason="Score is below the quality gate.",
    )
    register_approved_model.return_value = registration

    result = finalize_model_candidate(
        training_result=training_result,
        evaluation_result=evaluation_result,
        promotion_policy=POLICY,
        config=CONFIG,
    )

    assert result.registration is registration
    assert result.challenger_assignment is None
    assert result.promotion is None
    assign_challenger.assert_not_called()
    evaluate_and_promote_candidate.assert_not_called()


@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "evaluate_and_promote_candidate"
)
@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "assign_challenger"
)
@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "register_approved_model"
)
def test_custom_artifact_path_is_forwarded(
    register_approved_model: MagicMock,
    assign_challenger: MagicMock,
    evaluate_and_promote_candidate: MagicMock,
) -> None:
    training_result = build_training_result()
    evaluation_result = build_approved_evaluation()
    registration = build_registration()

    register_approved_model.return_value = registration
    assign_challenger.return_value = MagicMock()
    evaluate_and_promote_candidate.return_value = MagicMock()

    finalize_model_candidate(
        training_result=training_result,
        evaluation_result=evaluation_result,
        promotion_policy=POLICY,
        config=CONFIG,
        artifact_path="trained/model",
    )

    register_approved_model.assert_called_once_with(
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=CONFIG,
        artifact_path="trained/model",
        model_uri=None,
    )

@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "finalize_model_candidate"
)
@patch(
    "mlops_sales_forecasting.tracking.lifecycle."
    "load_promotion_policy"
)
def test_configured_lifecycle_loads_policy(
    load_promotion_policy: MagicMock,
    finalize_model_candidate: MagicMock,
) -> None:
    training_result = build_training_result()
    evaluation_result = build_approved_evaluation()
    lifecycle_result = MagicMock()

    load_promotion_policy.return_value = POLICY
    finalize_model_candidate.return_value = (
        lifecycle_result
    )

    result = finalize_configured_model_candidate(
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=CONFIG,
        artifact_path="trained/model",
        logged_model_uri="models:/m-logged-model",
    )

    assert result is lifecycle_result
    load_promotion_policy.assert_called_once_with(
        CONFIG
    )
    finalize_model_candidate.assert_called_once_with(
        training_result=training_result,
        evaluation_result=evaluation_result,
        promotion_policy=POLICY,
        config=CONFIG,
        artifact_path="trained/model",
        logged_model_uri="models:/m-logged-model",
    )