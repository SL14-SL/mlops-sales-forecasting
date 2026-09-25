import re
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
_SUPPORTED_SCHEMA_VERSIONS = {1}


class TaskType(StrEnum):
    """Machine-learning task supported by a serving release."""

    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"


@dataclass(frozen=True)
class ArtifactReference:
    """Reference to an immutable artifact within a serving release."""

    path: str
    sha256: str


@dataclass(frozen=True)
class ModelReference:
    """Reference to a registered machine-learning model."""

    name: str
    version: str
    run_id: str
    uri: str
    model_type: str


@dataclass(frozen=True)
class ServingReleaseManifest:
    """Persistent description of one complete serving release."""

    schema_version: int
    release_id: str
    created_at_utc: str
    task_type: TaskType
    model: ModelReference
    artifacts: dict[str, ArtifactReference]
    dataset_version: str | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest into a serializable dictionary."""
        return asdict(self)


def _require_non_empty_string(
    value: object,
    *,
    field_name: str,
) -> None:
    """Raise ValueError when a required string is empty or invalid."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Serving manifest has an invalid {field_name}."
        )


def validate_artifact_reference(
    reference: ArtifactReference,
    *,
    name: str,
) -> None:
    """Validate one immutable serving artifact reference."""
    if not isinstance(reference, ArtifactReference):
        raise ValueError(
            f"Serving manifest has an invalid {name} reference."
        )

    _require_non_empty_string(
        reference.path,
        field_name=f"{name} path",
    )

    if (
        reference.path.startswith("/")
        or reference.path.startswith("gs://")
        or ".." in reference.path.split("/")
    ):
        raise ValueError(
            f"Serving manifest {name} path must be relative."
        )

    if not _SHA256_PATTERN.fullmatch(reference.sha256):
        raise ValueError(
            f"Serving manifest {name} has an invalid SHA-256 checksum."
        )


def validate_model_reference(model: ModelReference) -> None:
    """Validate a registered-model reference."""
    if not isinstance(model, ModelReference):
        raise ValueError(
            "Serving manifest has an invalid model reference."
        )

    required_fields = {
        "model name": model.name,
        "model version": model.version,
        "model run ID": model.run_id,
        "model URI": model.uri,
        "model type": model.model_type,
    }

    for field_name, value in required_fields.items():
        _require_non_empty_string(
            value,
            field_name=field_name,
        )


def validate_serving_manifest(
    manifest: ServingReleaseManifest,
) -> None:
    """Raise ValueError when a serving manifest is incomplete."""
    if not isinstance(manifest, ServingReleaseManifest):
        raise ValueError(
            "Invalid serving manifest."
        )

    if manifest.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported serving manifest schema version: "
            f"{manifest.schema_version}."
        )

    _require_non_empty_string(
        manifest.release_id,
        field_name="release ID",
    )
    _require_non_empty_string(
        manifest.created_at_utc,
        field_name="creation timestamp",
    )

    try:
        timestamp = datetime.fromisoformat(
            manifest.created_at_utc.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            "Serving manifest has an invalid creation timestamp."
        ) from exc

    if timestamp.tzinfo is None:
        raise ValueError(
            "Serving manifest creation timestamp must include a timezone."
        )

    if not isinstance(manifest.task_type, TaskType):
        raise ValueError(
            "Serving manifest has an invalid task type."
        )

    validate_model_reference(manifest.model)

    if not isinstance(manifest.artifacts, dict):
        raise ValueError(
            "Serving manifest has invalid artifacts."
        )

    if not manifest.artifacts:
        raise ValueError(
            "Serving manifest contains no artifacts."
        )

    for name, reference in manifest.artifacts.items():
        _require_non_empty_string(
            name,
            field_name="artifact name",
        )
        validate_artifact_reference(
            reference,
            name=name,
        )

    if (
        manifest.metadata is not None
        and not isinstance(manifest.metadata, dict)
    ):
        raise ValueError(
            "Serving manifest metadata must be a mapping."
        )