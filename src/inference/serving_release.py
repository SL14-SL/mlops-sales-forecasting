from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, BinaryIO

import fsspec

from src.configs.loader import (
    file_exists,
    join_uri,
    read_text,
    remove_file,
    write_text,
    list_files,
)
from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingReleaseManifest,
)


MANIFEST_FILE_NAME = "serving_manifest.json"
ACTIVE_RELEASE_FILE_NAME = (
    "active_serving_release.json"
)
RELEASES_DIRECTORY_NAME = "serving_releases"

COPY_CHUNK_SIZE = 1024 * 1024


def build_release_id(
    model_version: str,
) -> str:
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    unique_suffix = uuid.uuid4().hex[:8]

    return (
        f"release-{timestamp}"
        f"-v{model_version}"
        f"-{unique_suffix}"
    )


def _copy_stream(
    source: BinaryIO,
    target: BinaryIO,
) -> None:
    while True:
        chunk = source.read(
            COPY_CHUNK_SIZE
        )

        if not chunk:
            break

        target.write(chunk)


def copy_uri(
    source_path: str,
    target_path: str,
) -> None:
    """
    Copy a file between any fsspec-supported locations.

    Supported combinations include:
    - local to local
    - local to GCS
    - GCS to local
    - GCS to GCS
    """
    if not file_exists(source_path):
        raise FileNotFoundError(
            f"Serving release source not found: "
            f"{source_path}"
        )

    with (
        fsspec.open(source_path, "rb") as source,
        fsspec.open(target_path, "wb") as target,
    ):
        _copy_stream(
            source,
            target,
        )


def sha256_uri(path: str) -> str:
    """Calculate SHA-256 for a local or remote artifact."""
    digest = hashlib.sha256()

    with fsspec.open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(
                COPY_CHUNK_SIZE
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: str,
    payload: dict[str, Any],
) -> None:
    write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
    )


def load_json(path: str) -> dict[str, Any]:
    payload = json.loads(
        read_text(path)
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object at: {path}"
        )

    return payload

def build_release_paths(
    *,
    models_path: str,
    release_id: str,
) -> dict[str, str]:
    release_root = join_uri(
        models_path,
        RELEASES_DIRECTORY_NAME,
        release_id,
    )

    return {
        "release_root": release_root,
        "manifest": join_uri(
            release_root,
            MANIFEST_FILE_NAME,
        ),
        "store_metadata": join_uri(
            release_root,
            "store.parquet",
        ),
        "store_state": join_uri(
            release_root,
            "latest_state.json",
        ),
        "known_calendar": join_uri(
            release_root,
            "known_calendar.parquet",
        ),
        "active_pointer": join_uri(
            models_path,
            ACTIVE_RELEASE_FILE_NAME,
        ),
        "prediction_probe": join_uri(
            release_root,
            "prediction_probe.json",
        ),
    }

def publish_serving_release(
    *,
    models_path: str,
    model_name: str,
    model_version: str,
    model_run_id: str,
    model_type: str,
    target_transformation: str,
    dataset_version: str | None,
    config_hash: str | None,
    git_commit: str | None,
    store_metadata_source: str,
    store_state_source: str,
    known_calendar_source: str,
    prediction_probe_payload: dict[str, Any],
) -> ServingReleaseManifest:
    """
    Publish a complete immutable serving release.

    Publication protocol:
    1. Copy artifacts into an immutable release prefix.
    2. Verify copied artifacts using SHA-256.
    3. Write the release manifest.
    4. Update the active pointer last.

    If any operation before step 4 fails, the previous active release remains
    selected.
    """
    resolved_model_version = str(
        model_version
    )

    release_id = build_release_id(
        resolved_model_version
    )

    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if file_exists(paths["manifest"]):
        raise FileExistsError(
            "Serving release already exists: "
            f"{release_id}"
        )

    artifact_sources = {
        "store_metadata": (
            store_metadata_source
        ),
        "store_state": (
            store_state_source
        ),
        "known_calendar": (
            known_calendar_source
        ),
    }

    artifact_references: dict[
        str,
        ServingArtifactReference,
    ] = {}

    probe_inputs = (
        prediction_probe_payload.get(
            "inputs"
        )
    )

    if (
        not isinstance(probe_inputs, list)
        or not probe_inputs
    ):
        raise ValueError(
            "Prediction probe payload must "
            "contain non-empty inputs."
        )

    try:
        for artifact_name, source_path in (
            artifact_sources.items()
        ):
            target_path = paths[artifact_name]

            source_hash = sha256_uri(
                source_path
            )

            copy_uri(
                source_path,
                target_path,
            )

            target_hash = sha256_uri(
                target_path
            )

            if target_hash != source_hash:
                raise ValueError(
                    "Serving artifact checksum mismatch "
                    f"after copy: {artifact_name}"
                )

            artifact_references[
                artifact_name
            ] = ServingArtifactReference(
                # Paths are relative to the release root.
                path={
                    "store_metadata": "store.parquet",
                    "store_state": "latest_state.json",
                    "known_calendar": (
                        "known_calendar.parquet"
                    ),
                }[artifact_name],
                sha256=target_hash,
            )
            write_json(
                paths["prediction_probe"],
                prediction_probe_payload,
            )

            prediction_probe_hash = sha256_uri(
                paths["prediction_probe"]
            )

            artifact_references[
                "prediction_probe"
            ] = ServingArtifactReference(
                path="prediction_probe.json",
                sha256=prediction_probe_hash,
            )


        manifest = ServingReleaseManifest(
            schema_version=2,
            release_id=release_id,
            created_at_utc=datetime.now(
                timezone.utc
            ).isoformat(),
            model_name=model_name,
            model_version=(
                resolved_model_version
            ),
            model_run_id=model_run_id,
            model_uri=(
                f"models:/{model_name}/"
                f"{resolved_model_version}"
            ),
            model_type=model_type,
            target_transformation=(
                target_transformation
            ),
            dataset_version=dataset_version,
            config_hash=config_hash,
            git_commit=git_commit,
            store_metadata=artifact_references[
                "store_metadata"
            ],
            store_state=artifact_references[
                "store_state"
            ],
            known_calendar=artifact_references[
                "known_calendar"
            ],
            prediction_probe=artifact_references[
                "prediction_probe"
            ],
        )

        # Manifest is written only after all artifacts were verified.
        write_json(
            paths["manifest"],
            manifest.to_dict(),
        )

        # Read-after-write validation before publishing the pointer.
        stored_manifest = load_json(
            paths["manifest"]
        )

        if (
            stored_manifest.get("release_id")
            != release_id
        ):
            raise ValueError(
                "Stored serving manifest failed "
                "read-after-write validation."
            )

        # Commit point: update pointer only after complete publication.
        previous_release_id = None

        if file_exists(
            paths["active_pointer"]
        ):
            previous_release_id = (
                load_active_release_id(
                    models_path=models_path,
                )
            )

        activate_release_pointer(
            models_path=models_path,
            release_id=release_id,
            operation="promotion",
            previous_release_id=(
                previous_release_id
            ),
        )

        return manifest

    except Exception:
        # The active pointer has not been changed yet.
        # Remove incomplete release objects where possible.
        for key in (
            "manifest",
            "store_metadata",
            "store_state",
            "known_calendar",
            "prediction_probe",
        ):
            remove_file(
                paths[key]
            )

        raise

def _parse_artifact_reference(
    payload: dict[str, Any],
    field_name: str,
) -> ServingArtifactReference:
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

def load_serving_manifest(
    *,
    models_path: str,
    release_id: str,
) -> tuple[
    ServingReleaseManifest,
    str,
]:
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if not file_exists(paths["manifest"]):
        raise FileNotFoundError(
            "Serving release manifest not found: "
            f"{paths['manifest']}"
        )

    manifest = parse_serving_manifest(
        load_json(paths["manifest"])
    )

    if manifest.release_id != release_id:
        raise ValueError(
            "Requested release ID does not "
            "match the manifest."
        )

    return (
        manifest,
        paths["release_root"],
    )


def load_active_serving_manifest(
    *,
    models_path: str,
) -> tuple[
    ServingReleaseManifest,
    str,
]:
    release_id = load_active_release_id(
        models_path=models_path,
    )

    return load_serving_manifest(
        models_path=models_path,
        release_id=release_id,
    )

def load_serving_release_manifest(
    *,
    models_path: str,
    release_id: str,
) -> ServingReleaseManifest:
    """
    Load one immutable serving release manifest by release ID.
    """
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    manifest_path = paths["manifest"]

    if not file_exists(manifest_path):
        raise FileNotFoundError(
            "Serving release manifest not found: "
            f"{manifest_path}"
        )

    return parse_serving_manifest(
        load_json(manifest_path)
    )

def resolve_release_artifact_uri(
    *,
    release_root: str,
    reference: ServingArtifactReference,
) -> str:
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

def load_active_release_id(
    *,
    models_path: str,
) -> str:
    pointer_path = join_uri(
        models_path,
        ACTIVE_RELEASE_FILE_NAME,
    )

    if not file_exists(pointer_path):
        raise FileNotFoundError(
            "Active serving release pointer "
            f"not found: {pointer_path}"
        )

    pointer = load_json(
        pointer_path
    )

    release_id = pointer.get(
        "release_id"
    )

    if not release_id:
        raise ValueError(
            "Active serving release pointer "
            "has no release_id."
        )

    return str(release_id)


def activate_release_pointer(
    *,
    models_path: str,
    release_id: str,
    operation: str = "activation",
    previous_release_id: str | None = None,
) -> None:
    """
    Update the active release pointer.

    The caller must validate the target release before calling this function.
    """
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if not file_exists(paths["manifest"]):
        raise FileNotFoundError(
            "Cannot activate release without manifest: "
            f"{paths['manifest']}"
        )

    write_json(
        paths["active_pointer"],
        {
            "schema_version": 1,
            "release_id": release_id,
            "previous_release_id": (
                previous_release_id
            ),
            "operation": operation,
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )

    # Confirm that storage returns the newly written pointer.
    stored_release_id = load_active_release_id(
        models_path=models_path,
    )

    if stored_release_id != release_id:
        raise RuntimeError(
            "Active release pointer verification failed: "
            f"expected={release_id}, "
            f"actual={stored_release_id}"
        )


def list_serving_release_manifests(
    *,
    models_path: str,
) -> list[ServingReleaseManifest]:
    pattern = join_uri(
        models_path,
        RELEASES_DIRECTORY_NAME,
        "*",
        MANIFEST_FILE_NAME,
    )

    manifest_paths = list_files(
        pattern
    )

    manifests: list[
        ServingReleaseManifest
    ] = []

    for manifest_path in manifest_paths:
        manifests.append(
            parse_serving_manifest(
                load_json(manifest_path)
            )
        )

    return sorted(
        manifests,
        key=lambda manifest: (
            manifest.created_at_utc
        ),
        reverse=True,
    )

