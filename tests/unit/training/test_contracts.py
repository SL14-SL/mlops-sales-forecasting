from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from mlops_sales_forecasting.data.contracts import (
    DatasetSplits,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    ModelEvaluator,
    ModelTrainer,
    TrainingResult,
)


def build_splits() -> DatasetSplits:
    return DatasetSplits(
        train=pd.DataFrame(
            {
                "feature": [1, 2],
                "target": [0, 1],
            }
        ),
        validation=pd.DataFrame(
            {
                "feature": [3],
                "target": [1],
            }
        ),
    )


def test_training_result_contains_model_metadata() -> None:
    model = object()

    result = TrainingResult(
        model=model,
        run_id="run-123",
        metrics={
            "validation_score": 0.82,
        },
        parameters={
            "max_depth": 5,
        },
        artifacts={
            "model": "artifacts/model.bin",
        },
    )

    assert result.model is model
    assert result.run_id == "run-123"
    assert result.metrics == {
        "validation_score": 0.82,
    }
    assert result.parameters == {
        "max_depth": 5,
    }
    assert result.artifacts == {
        "model": "artifacts/model.bin",
    }


def test_training_result_requires_model() -> None:
    with pytest.raises(
        ValueError,
        match="must contain a model",
    ):
        TrainingResult(
            model=None,
            run_id="run-123",
            metrics={},
        )


@pytest.mark.parametrize(
    "run_id",
    ["", 7, None],
)
def test_training_result_requires_run_id(
    run_id: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="must contain a run ID",
    ):
        TrainingResult(
            model=object(),
            run_id=run_id,  # type: ignore[arg-type]
            metrics={},
        )


@pytest.mark.parametrize(
    "metrics",
    [
        {"": 0.5},
        {7: 0.5},
    ],
)
def test_metric_names_must_be_valid(
    metrics: dict[object, float],
) -> None:
    with pytest.raises(
        ValueError,
        match="Metric names",
    ):
        TrainingResult(
            model=object(),
            run_id="run-123",
            metrics=metrics,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    ["invalid", True, None],
)
def test_metrics_must_be_numeric(
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        TrainingResult(
            model=object(),
            run_id="run-123",
            metrics={
                "score": value,
            },  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_metrics_must_be_finite(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        TrainingResult(
            model=object(),
            run_id="run-123",
            metrics={
                "score": value,
            },
        )


@pytest.mark.parametrize(
    ("name", "uri"),
    [
        ("", "artifacts/model.bin"),
        ("model", ""),
        ("model", None),
    ],
)
def test_artifacts_require_names_and_uris(
    name: object,
    uri: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Artifact",
    ):
        TrainingResult(
            model=object(),
            run_id="run-123",
            metrics={},
            artifacts={
                name: uri,
            },  # type: ignore[dict-item]
        )


def test_approved_evaluation_accepts_no_reasons() -> None:
    result = EvaluationResult(
        metrics={
            "validation_score": 0.9,
        },
        approved=True,
    )

    assert result.approved is True
    assert result.reasons == ()


def test_rejected_evaluation_requires_reason() -> None:
    with pytest.raises(
        ValueError,
        match="must contain a reason",
    ):
        EvaluationResult(
            metrics={
                "validation_score": 0.2,
            },
            approved=False,
        )


def test_rejected_evaluation_accepts_reasons() -> None:
    result = EvaluationResult(
        metrics={
            "validation_score": 0.2,
        },
        approved=False,
        reasons=(
            "Validation score is below threshold.",
        ),
    )

    assert result.approved is False
    assert result.reasons == (
        "Validation score is below threshold.",
    )


@pytest.mark.parametrize(
    "reason",
    ["", 7, None],
)
def test_evaluation_reasons_must_be_valid(
    reason: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="reasons",
    ):
        EvaluationResult(
            metrics={},
            approved=False,
            reasons=(
                reason,
            ),  # type: ignore[arg-type]
        )


def test_evaluation_approval_must_be_boolean() -> None:
    with pytest.raises(
        TypeError,
        match="must be a boolean",
    ):
        EvaluationResult(
            metrics={},
            approved=1,  # type: ignore[arg-type]
        )


class ExampleTrainer:
    def train(
        self,
        datasets: DatasetSplits,
        config: Mapping[str, Any],
    ) -> TrainingResult:
        return TrainingResult(
            model={
                "training_rows": len(
                    datasets.train
                ),
            },
            run_id=str(config["run_id"]),
            metrics={
                "training_score": 0.8,
            },
        )


class ExampleEvaluator:
    def evaluate(
        self,
        training_result: TrainingResult,
        datasets: DatasetSplits,
        config: Mapping[str, Any],
    ) -> EvaluationResult:
        score = float(
            training_result.metrics[
                "training_score"
            ]
        )
        threshold = float(
            config["minimum_score"]
        )

        return EvaluationResult(
            metrics={
                "validation_score": score,
                "validation_rows": float(
                    len(datasets.validation)
                ),
            },
            approved=score >= threshold,
            reasons=(
                ()
                if score >= threshold
                else (
                    "Validation score is below threshold.",
                )
            ),
        )


def test_structural_training_protocols() -> None:
    assert isinstance(
        ExampleTrainer(),
        ModelTrainer,
    )
    assert isinstance(
        ExampleEvaluator(),
        ModelEvaluator,
    )


def test_protocol_implementations_form_pipeline() -> None:
    datasets = build_splits()

    training_result = ExampleTrainer().train(
        datasets,
        {
            "run_id": "run-456",
        },
    )
    evaluation_result = ExampleEvaluator().evaluate(
        training_result,
        datasets,
        {
            "minimum_score": 0.7,
        },
    )

    assert training_result.run_id == "run-456"
    assert evaluation_result.approved is True
    assert evaluation_result.metrics[
        "validation_rows"
    ] == 1.0