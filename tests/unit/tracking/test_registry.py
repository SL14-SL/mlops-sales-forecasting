from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.tracking import registry
from mlops_sales_forecasting.tracking.registry import (
    ModelRegistrationResult,
    register_approved_model,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


def build_config() -> dict:
    return {
        "tracking": {
            "model_name": "example-model-dev",
        },
    }


def build_training_result() -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id="mlflow-run-123",
        metrics={
            "validation_score": 0.82,
        },
    )


def build_evaluation_result(
    *,
    approved: bool,
) -> EvaluationResult:
    return EvaluationResult(
        metrics={
            "validation_score": 0.82,
        },
        approved=approved,
        reasons=(
            ()
            if approved
            else (
                "Candidate did not improve baseline.",
            )
        ),
    )


def test_approved_model_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_version = MagicMock()
    registered_version.version = "7"

    register_model = MagicMock(
        return_value=registered_version
    )
    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        register_model,
    )

    result = register_approved_model(
        training_result=build_training_result(),
        evaluation_result=(
            build_evaluation_result(
                approved=True
            )
        ),
        config=build_config(),
    )

    assert isinstance(
        result,
        ModelRegistrationResult,
    )
    assert result.registered is True
    assert result.run_id == "mlflow-run-123"
    assert result.model_name == (
        "example-model-dev"
    )
    assert result.model_version == "7"
    assert result.model_uri == (
        "models:/example-model-dev/7"
    )
    assert result.reason is None

    register_model.assert_called_once_with(
        model_uri="runs:/mlflow-run-123/model",
        name="example-model-dev",
    )


def test_custom_artifact_path_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_version = MagicMock()
    registered_version.version = "3"

    register_model = MagicMock(
        return_value=registered_version
    )
    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        register_model,
    )

    register_approved_model(
        training_result=build_training_result(),
        evaluation_result=(
            build_evaluation_result(
                approved=True
            )
        ),
        config=build_config(),
        artifact_path="candidate/model",
    )

    register_model.assert_called_once_with(
        model_uri=(
            "runs:/mlflow-run-123/"
            "candidate/model"
        ),
        name="example-model-dev",
    )


def test_rejected_model_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_model = MagicMock()
    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        register_model,
    )

    result = register_approved_model(
        training_result=build_training_result(),
        evaluation_result=(
            build_evaluation_result(
                approved=False
            )
        ),
        config=build_config(),
    )

    assert result.registered is False
    assert result.model_version is None
    assert result.model_uri is None
    assert result.reason == (
        "Candidate did not improve baseline."
    )
    register_model.assert_not_called()


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
        7,
        None,
        "${MODEL_NAME}",
    ],
)
def test_model_name_must_be_resolved(
    model_name: object,
) -> None:
    config = build_config()
    config["tracking"]["model_name"] = (
        model_name
    )

    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        register_approved_model(
            training_result=(
                build_training_result()
            ),
            evaluation_result=(
                build_evaluation_result(
                    approved=True
                )
            ),
            config=config,
        )


@pytest.mark.parametrize(
    "artifact_path",
    [
        "",
        "   ",
        "/absolute/model",
        "../outside",
        "models/../../outside",
    ],
)
def test_model_artifact_path_must_be_safe(
    artifact_path: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="artifact path",
    ):
        register_approved_model(
            training_result=(
                build_training_result()
            ),
            evaluation_result=(
                build_evaluation_result(
                    approved=True
                )
            ),
            config=build_config(),
            artifact_path=artifact_path,
        )


def test_registration_requires_training_result() -> None:
    with pytest.raises(
        TypeError,
        match="requires TrainingResult",
    ):
        register_approved_model(
            training_result=object(),  # type: ignore[arg-type]
            evaluation_result=(
                build_evaluation_result(
                    approved=True
                )
            ),
            config=build_config(),
        )


def test_registration_requires_evaluation_result() -> None:
    with pytest.raises(
        TypeError,
        match="requires EvaluationResult",
    ):
        register_approved_model(
            training_result=(
                build_training_result()
            ),
            evaluation_result=object(),  # type: ignore[arg-type]
            config=build_config(),
        )