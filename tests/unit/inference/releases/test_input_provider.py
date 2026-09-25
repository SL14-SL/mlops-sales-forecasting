from collections.abc import Mapping
from typing import Any

import pytest

from mlops_sales_forecasting.inference.releases.artifact_publisher import (
    ServingArtifactSource,
)
from mlops_sales_forecasting.inference.releases.contracts import (
    TaskType,
)
from mlops_sales_forecasting.inference.releases.input_provider import (
    ServingReleaseInput,
    build_serving_release_input,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


class RecordingInputProvider:
    def __init__(
        self,
        release_input: ServingReleaseInput,
    ) -> None:
        self.release_input = release_input
        self.calls: list[
            tuple[
                TrainingResult,
                EvaluationResult,
                Mapping[str, Any],
            ]
        ] = []

    def build_release_input(
        self,
        *,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
        config: Mapping[str, Any],
    ) -> ServingReleaseInput:
        self.calls.append(
            (
                training_result,
                evaluation_result,
                config,
            )
        )

        return self.release_input


class InvalidProvider:
    pass


class InvalidResultProvider:
    def build_release_input(
        self,
        *,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
        config: Mapping[str, Any],
    ) -> object:
        del (
            training_result,
            evaluation_result,
            config,
        )

        return object()


def build_training_result() -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id="run-7",
        metrics={"score": 0.90},
    )


def build_evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        metrics={"score": 0.90},
        approved=True,
    )


def build_release_input() -> ServingReleaseInput:
    return ServingReleaseInput(
        task_type=TaskType.CLASSIFICATION,
        model_type="xgboost",
        sources={
            "feature_schema": (
                ServingArtifactSource(
                    source_uri=(
                        "artifacts/feature_schema.json"
                    ),
                    relative_path=(
                        "feature_schema.json"
                    ),
                )
            )
        },
        metadata={
            "decision_threshold": 0.42,
        },
        dataset_version="dataset-v3",
        config_hash="config-hash",
        git_commit="abc123",
    )


def test_builds_release_input_through_provider() -> None:
    training_result = build_training_result()
    evaluation_result = (
        build_evaluation_result()
    )
    config = {
        "environment": "dev",
    }
    expected = build_release_input()
    provider = RecordingInputProvider(
        expected
    )

    result = build_serving_release_input(
        provider=provider,
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=config,
    )

    assert result is expected
    assert provider.calls == [
        (
            training_result,
            evaluation_result,
            config,
        )
    ]


def test_requires_provider_protocol() -> None:
    with pytest.raises(
        TypeError,
        match="must implement",
    ):
        build_serving_release_input(
            provider=InvalidProvider(),
            training_result=(
                build_training_result()
            ),
            evaluation_result=(
                build_evaluation_result()
            ),
            config={},
        )


def test_rejects_invalid_provider_result() -> None:
    with pytest.raises(
        TypeError,
        match="must return ServingReleaseInput",
    ):
        build_serving_release_input(
            provider=InvalidResultProvider(),
            training_result=(
                build_training_result()
            ),
            evaluation_result=(
                build_evaluation_result()
            ),
            config={},
        )


def test_release_input_requires_sources() -> None:
    with pytest.raises(
        ValueError,
        match="at least one artifact source",
    ):
        ServingReleaseInput(
            task_type=TaskType.CLASSIFICATION,
            model_type="xgboost",
            sources={},
        )


def test_release_input_requires_model_type() -> None:
    with pytest.raises(
        ValueError,
        match="model type",
    ):
        ServingReleaseInput(
            task_type=TaskType.CLASSIFICATION,
            model_type="",
            sources={
                "schema": (
                    ServingArtifactSource(
                        source_uri="schema.json",
                        relative_path="schema.json",
                    )
                )
            },
        )


def test_release_input_requires_task_type() -> None:
    with pytest.raises(
        TypeError,
        match="requires TaskType",
    ):
        ServingReleaseInput(
            task_type="classification",
            model_type="xgboost",
            sources={
                "schema": (
                    ServingArtifactSource(
                        source_uri="schema.json",
                        relative_path="schema.json",
                    )
                )
            },
        )


def test_release_input_requires_valid_sources() -> None:
    with pytest.raises(
        TypeError,
        match="ServingArtifactSource",
    ):
        ServingReleaseInput(
            task_type=TaskType.CLASSIFICATION,
            model_type="xgboost",
            sources={
                "schema": object(),
            },
        )