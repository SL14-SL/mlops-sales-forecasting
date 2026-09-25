from datetime import datetime

from ...configs.paths import join_uri
from ...storage.filesystem import file_exists, list_files
from .contracts import (
    ServingReleaseManifest,
    TaskType,
)
from .manifest import read_serving_manifest
from .pointer import (
    ActiveReleasePointer,
    ReleaseOperation,
    activate_release_pointer,
    load_active_release_pointer,
)
from .storage import (
    MANIFEST_FILE_NAME,
    RELEASES_DIRECTORY_NAME,
    build_release_paths,
)


def load_release_manifest(
    *,
    models_path: str,
    release_id: str,
) -> tuple[ServingReleaseManifest, str]:
    """Load a validated manifest and return its release root."""
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )
    manifest_path = paths["manifest"]

    if not file_exists(manifest_path):
        raise FileNotFoundError(
            f"Serving release manifest not found: {manifest_path}"
        )

    manifest = read_serving_manifest(manifest_path)

    if manifest.release_id != release_id:
        raise ValueError(
            "Serving manifest release ID does not match "
            f"its storage location: expected={release_id}, "
            f"actual={manifest.release_id}."
        )

    return manifest, paths["release_root"]


def load_active_release_manifest(
    *,
    models_path: str,
) -> tuple[ServingReleaseManifest, str]:
    """Load the manifest referenced by the active pointer."""
    pointer = load_active_release_pointer(
        models_path=models_path
    )

    return load_release_manifest(
        models_path=models_path,
        release_id=pointer.release_id,
    )


def _manifest_timestamp(
    manifest: ServingReleaseManifest,
) -> datetime:
    """Return the parsed release creation timestamp."""
    return datetime.fromisoformat(
        manifest.created_at_utc.replace("Z", "+00:00")
    )


def list_release_manifests(
    *,
    models_path: str,
) -> list[ServingReleaseManifest]:
    """List valid serving manifests from newest to oldest."""
    pattern = join_uri(
        models_path,
        RELEASES_DIRECTORY_NAME,
        "*",
        MANIFEST_FILE_NAME,
    )

    manifests = [
        read_serving_manifest(manifest_path)
        for manifest_path in list_files(pattern)
    ]

    return sorted(
        manifests,
        key=_manifest_timestamp,
        reverse=True,
    )


def activate_release(
    *,
    models_path: str,
    release_id: str,
    operation: ReleaseOperation = ReleaseOperation.ACTIVATION,
    expected_task_type: TaskType | None = None,
    previous_release_id: str | None = None,
    updated_at_utc: str | None = None,
) -> ActiveReleasePointer:
    """Validate and activate a serving release."""
    manifest, _ = load_release_manifest(
        models_path=models_path,
        release_id=release_id,
    )

    if (
        expected_task_type is not None
        and manifest.task_type is not expected_task_type
    ):
        raise ValueError(
            "Serving release task type does not match project: "
            f"expected={expected_task_type.value}, "
            f"actual={manifest.task_type.value}."
        )

    return activate_release_pointer(
        models_path=models_path,
        release_id=release_id,
        operation=operation,
        previous_release_id=previous_release_id,
        updated_at_utc=updated_at_utc,
    )


def rollback_active_release(
    *,
    models_path: str,
    expected_task_type: TaskType | None = None,
    updated_at_utc: str | None = None,
) -> ActiveReleasePointer:
    """Roll back to the release stored as the previous release."""
    current_pointer = load_active_release_pointer(
        models_path=models_path
    )
    rollback_release_id = current_pointer.previous_release_id

    if rollback_release_id is None:
        raise ValueError(
            "Active release pointer contains no rollback candidate."
        )

    return activate_release(
        models_path=models_path,
        release_id=rollback_release_id,
        operation=ReleaseOperation.ROLLBACK,
        expected_task_type=expected_task_type,
        previous_release_id=current_pointer.release_id,
        updated_at_utc=updated_at_utc,
    )