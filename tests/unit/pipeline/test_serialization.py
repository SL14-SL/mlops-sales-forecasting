from datetime import UTC, datetime

import pytest

from mlops_sales_forecasting.pipeline.serialization import (
    PIPELINE_RUN_SCHEMA_VERSION,
    parse_pipeline_run,
    pipeline_run_to_dict,
)
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
COMPLETED_AT = datetime(
    2026,
    9,
    15,
    8,
    1,
    tzinfo=UTC,
)


def test_running_pipeline_round_trip() -> None:
    original = PipelineRun.start(
        "pipeline-run-123",
        started_at_utc=STARTED_AT,
    )

    payload = pipeline_run_to_dict(original)
    restored = parse_pipeline_run(payload)

    assert restored == original
    assert payload["schema_version"] == (
        PIPELINE_RUN_SCHEMA_VERSION
    )


def test_successful_pipeline_round_trip() -> None:
    original = PipelineRun.start(
        "pipeline-run-123",
        started_at_utc=STARTED_AT,
    ).succeed(
        completed_at_utc=COMPLETED_AT
    )

    assert parse_pipeline_run(
        pipeline_run_to_dict(original)
    ) == original


def test_rejected_pipeline_round_trip() -> None:
    original = PipelineRun.start(
        "pipeline-run-123",
        started_at_utc=STARTED_AT,
    ).reject(
        ("Candidate did not pass evaluation.",),
        completed_at_utc=COMPLETED_AT,
    )

    restored = parse_pipeline_run(
        pipeline_run_to_dict(original)
    )

    assert restored == original
    assert restored.status is (
        PipelineRunStatus.REJECTED
    )


def test_failed_pipeline_round_trip() -> None:
    original = PipelineRun.start(
        "pipeline-run-123",
        started_at_utc=STARTED_AT,
    ).fail(
        RuntimeError("Training failed."),
        completed_at_utc=COMPLETED_AT,
    )

    restored = parse_pipeline_run(
        pipeline_run_to_dict(original)
    )

    assert restored == original
    assert restored.error_type == "RuntimeError"
    assert restored.error_message == (
        "Training failed."
    )


def test_parser_accepts_utc_z_suffix() -> None:
    payload = pipeline_run_to_dict(
        PipelineRun.start(
            "pipeline-run-123",
            started_at_utc=STARTED_AT,
        )
    )
    payload["started_at_utc"] = (
        "2026-09-15T08:00:00Z"
    )

    restored = parse_pipeline_run(payload)

    assert restored.started_at_utc == STARTED_AT


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "run_id",
        "status",
        "started_at_utc",
    ],
)
def test_required_fields_must_be_present(
    field: str,
) -> None:
    payload = pipeline_run_to_dict(
        PipelineRun.start(
            "pipeline-run-123",
            started_at_utc=STARTED_AT,
        )
    )
    del payload[field]

    with pytest.raises(
        ValueError,
        match="missing required field",
    ):
        parse_pipeline_run(payload)


@pytest.mark.parametrize(
    "schema_version",
    [
        2,
        "1",
        True,
        None,
    ],
)
def test_schema_version_must_be_supported(
    schema_version: object,
) -> None:
    payload = pipeline_run_to_dict(
        PipelineRun.start(
            "pipeline-run-123",
            started_at_utc=STARTED_AT,
        )
    )
    payload["schema_version"] = schema_version

    with pytest.raises(
        ValueError,
        match="schema version",
    ):
        parse_pipeline_run(payload)


def test_status_must_be_supported() -> None:
    payload = pipeline_run_to_dict(
        PipelineRun.start(
            "pipeline-run-123",
            started_at_utc=STARTED_AT,
        )
    )
    payload["status"] = "unknown"

    with pytest.raises(
        ValueError,
        match="status is unsupported",
    ):
        parse_pipeline_run(payload)


@pytest.mark.parametrize(
    "timestamp",
    [
        "",
        "not-a-timestamp",
        7,
    ],
)
def test_start_timestamp_must_be_valid(
    timestamp: object,
) -> None:
    payload = pipeline_run_to_dict(
        PipelineRun.start(
            "pipeline-run-123",
            started_at_utc=STARTED_AT,
        )
    )
    payload["started_at_utc"] = timestamp

    with pytest.raises(
        ValueError,
        match="ISO timestamp",
    ):
        parse_pipeline_run(payload)


def test_rejection_reasons_must_be_list() -> None:
    payload = pipeline_run_to_dict(
        PipelineRun.start(
            "pipeline-run-123",
            started_at_utc=STARTED_AT,
        )
    )
    payload["rejection_reasons"] = (
        "Candidate rejected.",
    )

    with pytest.raises(
        ValueError,
        match="must be a list",
    ):
        parse_pipeline_run(payload)


@pytest.mark.parametrize(
    "field",
    [
        "error_type",
        "error_message",
    ],
)
def test_optional_error_fields_must_be_strings(
    field: str,
) -> None:
    payload = pipeline_run_to_dict(
        PipelineRun.start(
            "pipeline-run-123",
            started_at_utc=STARTED_AT,
        )
    )
    payload[field] = 7

    with pytest.raises(
        ValueError,
        match=field,
    ):
        parse_pipeline_run(payload)


def test_payload_must_be_mapping() -> None:
    with pytest.raises(
        TypeError,
        match="must be a mapping",
    ):
        parse_pipeline_run(
            []  # type: ignore[arg-type]
        )