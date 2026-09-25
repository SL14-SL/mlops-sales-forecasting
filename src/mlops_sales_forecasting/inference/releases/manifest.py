import json
from pathlib import Path
from typing import Any

from ...storage.filesystem import read_text, write_text
from .contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
    validate_serving_manifest,
)


def _required_value(
    payload: dict[str, Any],
    field_name: str,
) -> Any:
    """Return a required manifest value."""
    if field_name not in payload:
        raise ValueError(
            f"Serving manifest is missing required field: {field_name}."
        )

    return payload[field_name]


def _optional_string(
    payload: dict[str, Any],
    field_name: str,
) -> str | None:
    """Return an optional manifest value as a string."""
    value = payload.get(field_name)

    if value is None:
        return None

    return str(value)


def parse_artifact_reference(
    payload: object,
    *,
    name: str,
) -> ArtifactReference:
    """Parse one serving-artifact reference."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid serving artifact reference: {name}."
        )

    path = _required_value(payload, "path")
    checksum = _required_value(payload, "sha256")

    return ArtifactReference(
        path=str(path),
        sha256=str(checksum),
    )


def parse_model_reference(
    payload: object,
) -> ModelReference:
    """Parse a registered-model reference."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Invalid serving model reference."
        )

    return ModelReference(
        name=str(_required_value(payload, "name")),
        version=str(_required_value(payload, "version")),
        run_id=str(_required_value(payload, "run_id")),
        uri=str(_required_value(payload, "uri")),
        model_type=str(_required_value(payload, "model_type")),
    )


def parse_serving_manifest(
    payload: object,
) -> ServingReleaseManifest:
    """Parse and validate a persisted serving manifest."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Serving manifest must contain a JSON object."
        )

    artifact_payload = _required_value(payload, "artifacts")

    if not isinstance(artifact_payload, dict):
        raise ValueError(
            "Serving manifest artifacts must contain a JSON object."
        )

    artifacts = {
        str(name): parse_artifact_reference(
            reference,
            name=str(name),
        )
        for name, reference in artifact_payload.items()
    }

    metadata = payload.get("metadata")

    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(
            "Serving manifest metadata must contain a JSON object."
        )

    try:
        task_type = TaskType(
            str(_required_value(payload, "task_type"))
        )
    except ValueError as exc:
        raise ValueError(
            "Serving manifest has an unsupported task type."
        ) from exc

    manifest = ServingReleaseManifest(
        schema_version=int(
            _required_value(payload, "schema_version")
        ),
        release_id=str(
            _required_value(payload, "release_id")
        ),
        created_at_utc=str(
            _required_value(payload, "created_at_utc")
        ),
        task_type=task_type,
        model=parse_model_reference(
            _required_value(payload, "model")
        ),
        artifacts=artifacts,
        dataset_version=_optional_string(
            payload,
            "dataset_version",
        ),
        config_hash=_optional_string(
            payload,
            "config_hash",
        ),
        git_commit=_optional_string(
            payload,
            "git_commit",
        ),
        metadata=metadata,
    )

    validate_serving_manifest(manifest)
    return manifest


def manifest_to_json(
    manifest: ServingReleaseManifest,
) -> str:
    """Serialize a validated serving manifest as formatted JSON."""
    validate_serving_manifest(manifest)

    return (
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_serving_manifest(
    path: str | Path,
    manifest: ServingReleaseManifest,
) -> None:
    """Validate and persist a serving manifest."""
    write_text(
        str(path),
        manifest_to_json(manifest),
    )


def read_serving_manifest(
    path: str | Path,
) -> ServingReleaseManifest:
    """Read, parse and validate a serving manifest."""
    serialized_manifest = read_text(str(path))

    try:
        payload = json.loads(serialized_manifest)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Serving manifest contains invalid JSON: {path}."
        ) from exc

    return parse_serving_manifest(payload)