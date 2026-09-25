from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.pipeline import runner
from mlops_sales_forecasting.pipeline.runner import (
    PipelineResult,
    TrackedPipelineResult,
    run_tracked_training_pipeline,
)
from mlops_sales_forecasting.pipeline.status import (
    PipelineRun,
    PipelineRunStatus,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
)

STARTED_AT = datetime(
    2026,
    9,
    15,
    8,
    0,
    tzinfo=UTC,
)
COMPLETED_AT = STARTED_AT + timedelta(
    seconds=15
)


def build_components() -> tuple[
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    return (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )


def build_pipeline_result(
    *,
    approved: bool,
) -> MagicMock:
    result = MagicMock(
        spec=PipelineResult
    )
    result.evaluation = EvaluationResult(
        metrics={
            "validation_score": 0.8,
        },
        approved=approved,
        reasons=(
            ()
            if approved
            else (
                "Candidate did not pass evaluation.",
            )
        ),
    )
    return result


def build_clock() -> MagicMock:
    return MagicMock(
        side_effect=[
            STARTED_AT,
            COMPLETED_AT,
        ]
    )


def test_successful_pipeline_reports_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = build_components()
    pipeline_result = build_pipeline_result(
        approved=True
    )
    observer = MagicMock()

    execute = MagicMock(
        return_value=pipeline_result
    )
    monkeypatch.setattr(
        runner,
        "run_training_pipeline",
        execute,
    )

    result = run_tracked_training_pipeline(
        ingestor=components[0],
        feature_builder=components[1],
        splitter=components[2],
        trainer=components[3],
        evaluator=components[4],
        config={
            "environment": "test",
        },
        run_id="pipeline-run-123",
        observer=observer,
        clock=build_clock(),
    )

    assert isinstance(
        result,
        TrackedPipelineResult,
    )
    assert result.pipeline is pipeline_result
    assert result.run.status is (
        PipelineRunStatus.SUCCEEDED
    )
    assert result.run.duration_seconds == 15

    observed_runs = [
        call.args[0]
        for call in observer.call_args_list
    ]

    assert [
        run.status
        for run in observed_runs
    ] == [
        PipelineRunStatus.RUNNING,
        PipelineRunStatus.SUCCEEDED,
    ]


def test_rejected_candidate_reports_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = build_components()
    pipeline_result = build_pipeline_result(
        approved=False
    )
    observer = MagicMock()

    monkeypatch.setattr(
        runner,
        "run_training_pipeline",
        MagicMock(
            return_value=pipeline_result
        ),
    )

    result = run_tracked_training_pipeline(
        ingestor=components[0],
        feature_builder=components[1],
        splitter=components[2],
        trainer=components[3],
        evaluator=components[4],
        config={},
        run_id="pipeline-run-456",
        observer=observer,
        clock=build_clock(),
    )

    assert result.run.status is (
        PipelineRunStatus.REJECTED
    )
    assert result.run.rejection_reasons == (
        "Candidate did not pass evaluation.",
    )

    final_run = observer.call_args_list[
        -1
    ].args[0]

    assert final_run.status is (
        PipelineRunStatus.REJECTED
    )


def test_failed_pipeline_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = build_components()
    observer = MagicMock()

    monkeypatch.setattr(
        runner,
        "run_training_pipeline",
        MagicMock(
            side_effect=RuntimeError(
                "Training failed."
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Training failed",
    ):
        run_tracked_training_pipeline(
            ingestor=components[0],
            feature_builder=components[1],
            splitter=components[2],
            trainer=components[3],
            evaluator=components[4],
            config={},
            run_id="pipeline-run-789",
            observer=observer,
            clock=build_clock(),
        )

    observed_runs: list[PipelineRun] = [
        call.args[0]
        for call in observer.call_args_list
    ]

    assert [
        run.status
        for run in observed_runs
    ] == [
        PipelineRunStatus.RUNNING,
        PipelineRunStatus.FAILED,
    ]

    failed_run = observed_runs[-1]

    assert failed_run.error_type == "RuntimeError"
    assert failed_run.error_message == (
        "Training failed."
    )
    assert failed_run.duration_seconds == 15


def test_pipeline_generates_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = build_components()
    observer = MagicMock()

    monkeypatch.setattr(
        runner,
        "run_training_pipeline",
        MagicMock(
            return_value=build_pipeline_result(
                approved=True
            )
        ),
    )

    result = run_tracked_training_pipeline(
        ingestor=components[0],
        feature_builder=components[1],
        splitter=components[2],
        trainer=components[3],
        evaluator=components[4],
        config={},
        observer=observer,
        clock=build_clock(),
    )

    assert result.run.run_id
    assert observer.call_args_list[
        0
    ].args[0].run_id == result.run.run_id


def test_tracked_runner_forwards_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = build_components()
    config = {
        "environment": "staging",
    }

    execute = MagicMock(
        return_value=build_pipeline_result(
            approved=True
        )
    )
    monkeypatch.setattr(
        runner,
        "run_training_pipeline",
        execute,
    )

    run_tracked_training_pipeline(
        ingestor=components[0],
        feature_builder=components[1],
        splitter=components[2],
        trainer=components[3],
        evaluator=components[4],
        config=config,
        run_id="pipeline-run-forwarding",
        clock=build_clock(),
    )

    execute.assert_called_once_with(
        ingestor=components[0],
        feature_builder=components[1],
        splitter=components[2],
        trainer=components[3],
        evaluator=components[4],
        config=config,
    )