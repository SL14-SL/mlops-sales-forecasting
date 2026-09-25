from datetime import UTC, datetime, timedelta

import pytest

from mlops_sales_forecasting.pipeline.status import (
    PipelineRun,
    PipelineRunStatus,
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
    seconds=12.5
)


def build_running_run() -> PipelineRun:
    return PipelineRun.start(
        "pipeline-run-123",
        started_at_utc=STARTED_AT,
    )


def test_start_creates_running_pipeline() -> None:
    run = build_running_run()

    assert run.run_id == "pipeline-run-123"
    assert run.status is PipelineRunStatus.RUNNING
    assert run.started_at_utc == STARTED_AT
    assert run.completed_at_utc is None
    assert run.duration_seconds is None


def test_pipeline_can_succeed() -> None:
    run = build_running_run().succeed(
        completed_at_utc=COMPLETED_AT
    )

    assert run.status is PipelineRunStatus.SUCCEEDED
    assert run.completed_at_utc == COMPLETED_AT
    assert run.duration_seconds == pytest.approx(
        12.5
    )
    assert run.rejection_reasons == ()
    assert run.error_type is None
    assert run.error_message is None


def test_pipeline_can_reject_candidate() -> None:
    run = build_running_run().reject(
        (
            "Candidate did not improve the baseline.",
        ),
        completed_at_utc=COMPLETED_AT,
    )

    assert run.status is PipelineRunStatus.REJECTED
    assert run.rejection_reasons == (
        "Candidate did not improve the baseline.",
    )
    assert run.duration_seconds == pytest.approx(
        12.5
    )


def test_pipeline_can_fail() -> None:
    run = build_running_run().fail(
        RuntimeError("Training failed."),
        completed_at_utc=COMPLETED_AT,
    )

    assert run.status is PipelineRunStatus.FAILED
    assert run.error_type == "RuntimeError"
    assert run.error_message == "Training failed."
    assert run.duration_seconds == pytest.approx(
        12.5
    )


@pytest.mark.parametrize(
    "run_id",
    ["", 7, None],
)
def test_pipeline_requires_run_id(
    run_id: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="run ID",
    ):
        PipelineRun.start(
            run_id,  # type: ignore[arg-type]
            started_at_utc=STARTED_AT,
        )


def test_start_timestamp_requires_timezone() -> None:
    with pytest.raises(
        ValueError,
        match="timezone",
    ):
        PipelineRun.start(
            "run-123",
            started_at_utc=datetime(
                2026,
                9,
                15,
                8,
                0,
            ),
        )


def test_completion_timestamp_requires_timezone() -> None:
    with pytest.raises(
        ValueError,
        match="timezone",
    ):
        build_running_run().succeed(
            completed_at_utc=datetime(
                2026,
                9,
                15,
                8,
                1,
            )
        )


def test_completion_cannot_precede_start() -> None:
    with pytest.raises(
        ValueError,
        match="cannot precede",
    ):
        build_running_run().succeed(
            completed_at_utc=(
                STARTED_AT - timedelta(seconds=1)
            )
        )


def test_rejected_pipeline_requires_reason() -> None:
    with pytest.raises(
        ValueError,
        match="at least one reason",
    ):
        build_running_run().reject(
            (),
            completed_at_utc=COMPLETED_AT,
        )


@pytest.mark.parametrize(
    "reason",
    ["", 7, None],
)
def test_rejection_reasons_must_be_strings(
    reason: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="rejection reasons",
    ):
        build_running_run().reject(
            (
                reason,
            ),  # type: ignore[arg-type]
            completed_at_utc=COMPLETED_AT,
        )


def test_failed_pipeline_requires_error_message() -> None:
    with pytest.raises(
        ValueError,
        match="error information",
    ):
        build_running_run().fail(
            RuntimeError(),
            completed_at_utc=COMPLETED_AT,
        )


@pytest.mark.parametrize(
    "terminal_status",
    [
        PipelineRunStatus.SUCCEEDED,
        PipelineRunStatus.REJECTED,
        PipelineRunStatus.FAILED,
    ],
)
def test_finished_pipeline_cannot_transition_again(
    terminal_status: PipelineRunStatus,
) -> None:
    running = build_running_run()

    if terminal_status is PipelineRunStatus.SUCCEEDED:
        finished = running.succeed(
            completed_at_utc=COMPLETED_AT
        )
    elif terminal_status is PipelineRunStatus.REJECTED:
        finished = running.reject(
            ("Candidate rejected.",),
            completed_at_utc=COMPLETED_AT,
        )
    else:
        finished = running.fail(
            RuntimeError("Failure."),
            completed_at_utc=COMPLETED_AT,
        )

    with pytest.raises(
        ValueError,
        match="Only a running pipeline",
    ):
        finished.succeed(
            completed_at_utc=(
                COMPLETED_AT + timedelta(seconds=1)
            )
        )