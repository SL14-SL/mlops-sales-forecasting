from collections.abc import Mapping
from typing import Any

import pytest

from mlops_sales_forecasting.tracking.model_artifact import (
    LoggedModelArtifact,
    log_model_artifact,
)
from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)


class RecordingModelLogger:
    def __init__(
        self,
        model_uri: str = (
            "models:/m-logged-model"
        ),
    ) -> None:
        self.model_uri = model_uri
        self.calls: list[
            tuple[
                TrainingResult,
                str,
                Mapping[str, Any],
            ]
        ] = []

    def log_model(
        self,
        training_result: TrainingResult,
        *,
        artifact_path: str,
        config: Mapping[str, Any],
    ) -> str:
        self.calls.append(
            (
                training_result,
                artifact_path,
                config,
            )
        )

        return self.model_uri


class InvalidLogger:
    pass

class EmptyUriLogger:
    def log_model(
        self,
        training_result: TrainingResult,
        *,
        artifact_path: str,
        config: Mapping[str, Any],
    ) -> str:
        del (
            training_result,
            artifact_path,
            config,
        )

        return ""


def build_training_result() -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id="run-123",
        metrics={"score": 0.91},
    )


def test_logs_model_through_adapter() -> None:
    logger = RecordingModelLogger()
    training_result = build_training_result()
    config = {
        "tracking": {
            "model_name": "example-model",
        }
    }

    result = log_model_artifact(
        logger=logger,
        training_result=training_result,
        config=config,
    )

    assert result == LoggedModelArtifact(
        run_id="run-123",
        artifact_path="model",
        model_uri="models:/m-logged-model",
    )
    assert logger.calls == [
        (
            training_result,
            "model",
            config,
        )
    ]


def test_supports_custom_artifact_path() -> None:
    logger = RecordingModelLogger()

    result = log_model_artifact(
        logger=logger,
        training_result=(
            build_training_result()
        ),
        config={},
        artifact_path="trained/model",
    )

    assert (
        result.model_uri
        == "models:/m-logged-model"
    )
    assert logger.calls[0][1] == "trained/model"


@pytest.mark.parametrize(
    "artifact_path",
    [
        "",
        "   ",
        "/model",
        "../model",
        "models/../model",
    ],
)
def test_rejects_invalid_artifact_path(
    artifact_path: str,
) -> None:
    logger = RecordingModelLogger()

    with pytest.raises(
        ValueError,
        match="artifact path",
    ):
        log_model_artifact(
            logger=logger,
            training_result=(
                build_training_result()
            ),
            config={},
            artifact_path=artifact_path,
        )

    assert logger.calls == []


def test_requires_model_artifact_logger() -> None:
    with pytest.raises(
        TypeError,
        match="must implement",
    ):
        log_model_artifact(
            logger=InvalidLogger(),
            training_result=(
                build_training_result()
            ),
            config={},
        )


def test_requires_training_result() -> None:
    logger = RecordingModelLogger()

    with pytest.raises(
        TypeError,
        match="requires TrainingResult",
    ):
        log_model_artifact(
            logger=logger,
            training_result=object(),
            config={},
        )

    assert logger.calls == []


def test_requires_logger_model_uri() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty MLflow model URI",
    ):
        log_model_artifact(
            logger=EmptyUriLogger(),
            training_result=(
                build_training_result()
            ),
            config={},
        )