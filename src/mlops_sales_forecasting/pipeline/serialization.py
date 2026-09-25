from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .status import PipelineRun, PipelineRunStatus

PIPELINE_RUN_SCHEMA_VERSION = 1


def pipeline_run_to_dict(
    run: PipelineRun,
) -> dict[str, Any]:
    """Serialize a pipeline run to a JSON-compatible dictionary."""
    return {
        "schema_version": (
            PIPELINE_RUN_SCHEMA_VERSION
        ),
        "run_id": run.run_id,
        "status": run.status.value,
        "started_at_utc": (
            run.started_at_utc.isoformat()
        ),
        "completed_at_utc": (
            run.completed_at_utc.isoformat()
            if run.completed_at_utc is not None
            else None
        ),
        "rejection_reasons": list(
            run.rejection_reasons
        ),
        "error_type": run.error_type,
        "error_message": run.error_message,
    }


def _required_value(
    payload: Mapping[str, Any],
    name: str,
) -> Any:
    try:
        return payload[name]
    except KeyError as exc:
        raise ValueError(
            f"Pipeline run is missing required field: {name}."
        ) from exc


def _parse_timestamp(
    value: object,
    *,
    name: str,
    required: bool,
) -> datetime | None:
    if value is None:
        if required:
            raise ValueError(
                f"Pipeline run field '{name}' is required."
            )

        return None

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Pipeline run field '{name}' "
            "must be an ISO timestamp."
        )

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"Pipeline run field '{name}' "
            "must be an ISO timestamp."
        ) from exc


def _optional_string(
    value: object,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Pipeline run field '{name}' "
            "must be a non-empty string or null."
        )

    return value


def parse_pipeline_run(
    payload: Mapping[str, Any],
) -> PipelineRun:
    """Parse and validate a serialized pipeline run."""
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Pipeline run payload must be a mapping."
        )

    schema_version = _required_value(
        payload,
        "schema_version",
    )

    if (
        isinstance(schema_version, bool)
        or schema_version
        != PIPELINE_RUN_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported pipeline run schema version."
        )

    run_id = _required_value(
        payload,
        "run_id",
    )
    status_value = _required_value(
        payload,
        "status",
    )

    if not isinstance(status_value, str):
        raise ValueError(
            "Pipeline run status must be a string."
        )

    try:
        status = PipelineRunStatus(
            status_value
        )
    except ValueError as exc:
        raise ValueError(
            "Pipeline run status is unsupported."
        ) from exc

    rejection_reasons = payload.get(
        "rejection_reasons",
        [],
    )

    if not isinstance(
        rejection_reasons,
        list,
    ):
        raise ValueError(
            "Pipeline rejection reasons must be a list."
        )

    started_at_utc = _parse_timestamp(
        _required_value(
            payload,
            "started_at_utc",
        ),
        name="started_at_utc",
        required=True,
    )

    if started_at_utc is None:
        raise ValueError(
            "Pipeline start timestamp is required."
        )

    return PipelineRun(
        run_id=run_id,
        status=status,
        started_at_utc=started_at_utc,
        completed_at_utc=_parse_timestamp(
            payload.get("completed_at_utc"),
            name="completed_at_utc",
            required=False,
        ),
        rejection_reasons=tuple(
            rejection_reasons
        ),
        error_type=_optional_string(
            payload.get("error_type"),
            name="error_type",
        ),
        error_message=_optional_string(
            payload.get("error_message"),
            name="error_message",
        ),
    )