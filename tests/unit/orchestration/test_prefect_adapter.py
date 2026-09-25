from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.orchestration import (
    prefect_adapter,
)
from mlops_sales_forecasting.pipeline.runner import (
    TrackedPipelineResult,
)
from mlops_sales_forecasting.pipeline.service import (
    TrainingPipeline,
)
from mlops_sales_forecasting.pipeline.status import (
    PipelineRun,
)


def build_pipeline_result(
    *,
    approved: bool,
) -> MagicMock:
    running = PipelineRun.start(
        "pipeline-run-123"
    )

    if approved:
        final_run = running.succeed()
    else:
        final_run = running.reject(
            ("Candidate did not pass evaluation.",)
        )

    result = MagicMock(
        spec=TrackedPipelineResult
    )
    result.run = final_run
    return result


def test_prefect_adapter_executes_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(
        spec=TrainingPipeline
    )
    expected_result = build_pipeline_result(
        approved=True
    )
    pipeline.run.return_value = expected_result

    logger = MagicMock()
    monkeypatch.setattr(
        prefect_adapter,
        "get_run_logger",
        MagicMock(return_value=logger),
    )

    result = (
        prefect_adapter
        .run_prefect_training_pipeline
        .fn(
            pipeline=pipeline,
            run_id="pipeline-run-123",
        )
    )

    assert result is expected_result
    pipeline.run.assert_called_once_with(
        run_id="pipeline-run-123"
    )
    logger.info.assert_called()


def test_prefect_adapter_allows_generated_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(
        spec=TrainingPipeline
    )
    pipeline.run.return_value = (
        build_pipeline_result(
            approved=True
        )
    )

    monkeypatch.setattr(
        prefect_adapter,
        "get_run_logger",
        MagicMock(
            return_value=MagicMock()
        ),
    )

    (
        prefect_adapter
        .run_prefect_training_pipeline
        .fn(
            pipeline=pipeline,
        )
    )

    pipeline.run.assert_called_once_with(
        run_id=None
    )


def test_prefect_adapter_logs_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(
        spec=TrainingPipeline
    )
    pipeline.run.return_value = (
        build_pipeline_result(
            approved=False
        )
    )

    logger = MagicMock()
    monkeypatch.setattr(
        prefect_adapter,
        "get_run_logger",
        MagicMock(return_value=logger),
    )

    result = (
        prefect_adapter
        .run_prefect_training_pipeline
        .fn(
            pipeline=pipeline,
            run_id="pipeline-run-123",
        )
    )

    assert result.run.rejection_reasons == (
        "Candidate did not pass evaluation.",
    )
    logger.warning.assert_called_once()


def test_prefect_adapter_logs_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(
        spec=TrainingPipeline
    )
    pipeline.run.side_effect = RuntimeError(
        "Training failed."
    )

    logger = MagicMock()
    monkeypatch.setattr(
        prefect_adapter,
        "get_run_logger",
        MagicMock(return_value=logger),
    )

    with pytest.raises(
        RuntimeError,
        match="Training failed",
    ):
        (
            prefect_adapter
            .run_prefect_training_pipeline
            .fn(
                pipeline=pipeline,
                run_id="pipeline-run-123",
            )
        )

    logger.exception.assert_called_once()