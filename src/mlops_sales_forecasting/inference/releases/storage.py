import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, BinaryIO

import fsspec

from ...configs.paths import join_uri
from ...storage.filesystem import (
    file_exists,
    read_text,
    write_text,
)
from .contracts import (
    ArtifactReference,
    validate_artifact_reference,
)

MANIFEST_FILE_NAME = "serving_manifest.json"
ACTIVE_RELEASE_FILE_NAME = "active_serving_release.json"
RELEASES_DIRECTORY_NAME = "serving_releases"
COPY_CHUNK_SIZE = 1024 * 1024

_RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_relative_path(path: str) -> None:
    """Validate a relative path contained within a release directory."""
    if (
        not path
        or path.startswith("/")
        or path.startswith("gs://")
        or "\\" in path
        or ".." in path.split("/")
    ):
        raise ValueError(
            f"Release artifact path must be relative: {path}"
        )

def validate_release_id(release_id: str) -> None:
    """Validate a serving-release identifier."""
    if not _RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError(
            f"Invalid serving release ID: {release_id}"
        )

def _copy_stream(
    source: BinaryIO,
    target: BinaryIO,
) -> None:
    """Copy binary content between two open file objects."""
    while chunk := source.read(COPY_CHUNK_SIZE):
        target.write(chunk)


def copy_uri(
    source_path: str,
    target_path: str,
) -> None:
    """Copy a file between fsspec-supported locations."""
    if not file_exists(source_path):
        raise FileNotFoundError(
            f"Serving release source not found: {source_path}"
        )

    with (
        fsspec.open(source_path, "rb") as source,
        fsspec.open(
            target_path,
            "wb",
            auto_mkdir=True,
        ) as target,
    ):
        _copy_stream(source, target)


def sha256_uri(path: str) -> str:
    """Calculate the SHA-256 checksum of a local or remote file."""
    digest = hashlib.sha256()

    with fsspec.open(path, "rb") as file_handle:
        while chunk := file_handle.read(COPY_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: str,
    payload: Mapping[str, Any],
) -> None:
    """Serialize a mapping as formatted JSON."""
    serialized_payload = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_text(path, serialized_payload)


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON object from a local or remote path."""
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON document at: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object at: {path}"
        )

    return payload


def build_release_paths(
    *,
    models_path: str,
    release_id: str,
    artifact_filenames: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build storage paths belonging to one immutable release."""
    
    validate_release_id(release_id)

    release_root = join_uri(
        models_path,
        RELEASES_DIRECTORY_NAME,
        release_id,
    )

    paths = {
        "release_root": release_root,
        "manifest": join_uri(
            release_root,
            MANIFEST_FILE_NAME,
        ),
        "active_pointer": join_uri(
            models_path,
            ACTIVE_RELEASE_FILE_NAME,
        ),
    }

    for name, filename in (artifact_filenames or {}).items():
        if not name:
            raise ValueError(
                "Release artifact name must not be empty."
            )

        if name in paths:
            raise ValueError(
                f"Duplicate release path name: {name}"
            )

        _validate_relative_path(filename)
        paths[name] = join_uri(
            release_root,
            filename,
        )

    return paths


def build_artifact_reference(
    *,
    relative_path: str,
    artifact_uri: str,
) -> ArtifactReference:
    """Create a checksummed reference to a release artifact."""
    _validate_relative_path(relative_path)

    if not file_exists(artifact_uri):
        raise FileNotFoundError(
            f"Serving artifact not found: {artifact_uri}"
        )

    reference = ArtifactReference(
        path=relative_path,
        sha256=sha256_uri(artifact_uri),
    )
    validate_artifact_reference(
        reference,
        name=relative_path,
    )
    return reference


def resolve_artifact_uri(
    *,
    release_root: str,
    reference: ArtifactReference,
) -> str:
    """Resolve and checksum-validate an artifact in a release."""
    validate_artifact_reference(
        reference,
        name=reference.path or "artifact",
    )

    artifact_uri = join_uri(
        release_root,
        reference.path,
    )

    if not file_exists(artifact_uri):
        raise FileNotFoundError(
            f"Serving artifact not found: {artifact_uri}"
        )

    actual_checksum = sha256_uri(artifact_uri)

    if actual_checksum != reference.sha256:
        raise ValueError(
            "Serving artifact checksum mismatch: "
            f"{reference.path}"
        )

    return artifact_uri