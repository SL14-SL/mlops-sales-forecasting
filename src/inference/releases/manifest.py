
from typing import Any

from src.configs.paths import join_uri
from src.storage.filesystem import file_exists

from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingReleaseManifest,
)

from src.inference.releases.storage import sha256_uri

def _parse_artifact_reference(
    payload: dict[str, Any],
    field_name: str,
) -> ServingArtifactReference:
    """
    Parse and validate one artifact reference from a manifest payload.
    """
    reference = payload.get(
        field_name
    )

    if not isinstance(reference, dict):
        raise ValueError(
            "Invalid serving artifact reference: "
            f"{field_name}"
        )

    path = reference.get("path")
    checksum = reference.get("sha256")

    if not path or not checksum:
        raise ValueError(
            "Incomplete serving artifact reference: "
            f"{field_name}"
        )

    return ServingArtifactReference(
        path=str(path),
        sha256=str(checksum),
    )


def parse_serving_manifest(
    payload: dict[str, Any],
) -> ServingReleaseManifest:
    """
    Parse and validate a serialized serving-release manifest.

    Args:
        payload: Untrusted manifest data loaded from release storage.

    Returns:
        A validated serving-release manifest.

    Raises:
        ValueError: If required fields, schema version or artifact references are
            missing or invalid.
    """
    schema_version = int(
        payload["schema_version"]
    )

    if schema_version not in {
        1,
        2,
    }:
        raise ValueError(
            "Unsupported serving manifest "
            f"schema version: {schema_version}"
        )

    prediction_probe = None

    if schema_version >= 2:
        prediction_probe = (
            _parse_artifact_reference(
                payload,
                "prediction_probe",
            )
        )

    manifest = ServingReleaseManifest(
        schema_version=schema_version,
        release_id=str(
            payload["release_id"]
        ),
        created_at_utc=str(
            payload["created_at_utc"]
        ),
        model_name=str(
            payload["model_name"]
        ),
        model_version=str(
            payload["model_version"]
        ),
        model_run_id=str(
            payload["model_run_id"]
        ),
        model_uri=str(
            payload["model_uri"]
        ),
        model_type=str(
            payload["model_type"]
        ),
        target_transformation=str(
            payload[
                "target_transformation"
            ]
        ),
        dataset_version=payload.get(
            "dataset_version"
        ),
        config_hash=payload.get(
            "config_hash"
        ),
        git_commit=payload.get(
            "git_commit"
        ),
        store_metadata=(
            _parse_artifact_reference(
                payload,
                "store_metadata",
            )
        ),
        store_state=(
            _parse_artifact_reference(
                payload,
                "store_state",
            )
        ),
        known_calendar=(
            _parse_artifact_reference(
                payload,
                "known_calendar",
            )
        ),
        prediction_probe=prediction_probe,
    )

    return manifest


def resolve_release_artifact_uri(
    *,
    release_root: str,
    reference: ServingArtifactReference,
) -> str:
    """
    Resolve an artifact reference against the immutable release directory.

    Absolute local and GCS references are preserved; relative references are
    resolved beneath the selected release.

    Raises:
        ValueError: If the reference would escape the release directory.
    """
    relative_path = reference.path

    if (
        relative_path.startswith("/")
        or relative_path.startswith("gs://")
        or ".." in relative_path.split("/")
    ):
        raise ValueError(
            "Serving artifact path must be "
            f"relative and contained: {relative_path}"
        )

    artifact_uri = join_uri(
        release_root,
        relative_path,
    )

    if not file_exists(artifact_uri):
        raise FileNotFoundError(
            f"Serving artifact not found: "
            f"{artifact_uri}"
        )

    actual_checksum = sha256_uri(
        artifact_uri
    )

    if actual_checksum != reference.sha256:
        raise ValueError(
            "Serving artifact checksum mismatch: "
            f"{relative_path}"
        )

    return artifact_uri

