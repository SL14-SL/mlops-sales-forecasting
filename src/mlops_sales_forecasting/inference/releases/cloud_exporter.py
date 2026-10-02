from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import mlflow

from ...configs.paths import join_uri
from ...storage.filesystem import (
    file_exists,
    remove_file,
)
from .artifact_publisher import (
    ServingArtifactSource,
    publish_serving_artifacts,
)
from .contracts import ServingReleaseManifest
from .manifest import write_serving_manifest
from .pointer import (
    ReleaseOperation,
    active_pointer_path,
)
from .publisher import PublishedServingRelease
from .repository import (
    activate_release,
    load_active_release_manifest,
    load_release_manifest,
)
from .storage import (
    build_release_paths,
    resolve_artifact_uri,
)

ModelDownloader = Callable[
    [str, str],
    str,
]


def _new_release_id() -> str:
    """Return a unique cloud release identifier."""
    return f"release-{uuid4()}"


def _download_model(
    model_uri: str,
    destination: str,
) -> str:
    """Download one MLflow model into a local directory."""
    return str(
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=destination,
        )
    )


def _model_sources(
    model_root: Path,
) -> dict[str, ServingArtifactSource]:
    """Build release sources for every model file."""
    if not model_root.is_dir():
        raise FileNotFoundError(f"Downloaded MLflow model directory does not exist: {model_root}")

    model_files = sorted(path for path in model_root.rglob("*") if path.is_file())

    if not model_files:
        raise ValueError("Downloaded MLflow model contains no files.")

    return {
        f"model_file_{index:04d}": (
            ServingArtifactSource(
                source_uri=str(path),
                relative_path=(f"model/{path.relative_to(model_root).as_posix()}"),
            )
        )
        for index, path in enumerate(
            model_files,
            start=1,
        )
    }


def _release_sources(
    *,
    manifest: ServingReleaseManifest,
    release_root: str,
) -> dict[str, ServingArtifactSource]:
    """Resolve and validate source release artifacts."""
    return {
        name: ServingArtifactSource(
            source_uri=resolve_artifact_uri(
                release_root=release_root,
                reference=reference,
            ),
            relative_path=reference.path,
        )
        for name, reference in manifest.artifacts.items()
        if not name.startswith("model_file_")
    }


def _cleanup_export(
    *,
    target_models_path: str,
    release_id: str,
    artifact_paths: list[str],
) -> None:
    """Best-effort cleanup of an incomplete export."""
    paths = build_release_paths(
        models_path=target_models_path,
        release_id=release_id,
    )

    for relative_path in reversed(artifact_paths):
        try:
            remove_file(
                join_uri(
                    paths["release_root"],
                    relative_path,
                )
            )
        except Exception:
            continue

    try:
        remove_file(paths["manifest"])
    except Exception:
        pass


def export_active_release(
    *,
    source_models_path: str,
    target_models_path: str,
    model_downloader: ModelDownloader = (_download_model),
    release_id: str | None = None,
) -> PublishedServingRelease:
    """
    Export the active release with a portable model.

    The active target pointer is written only after every
    release artifact and the manifest have been persisted.
    """
    if source_models_path == target_models_path:
        raise ValueError("Source and target models paths must differ.")

    source_manifest, source_release_root = load_active_release_manifest(
        models_path=source_models_path,
    )

    resolved_release_id = release_id or _new_release_id()
    target_paths = build_release_paths(
        models_path=target_models_path,
        release_id=resolved_release_id,
    )

    if file_exists(target_paths["manifest"]):
        raise FileExistsError(f"Target serving release already exists: {resolved_release_id}")

    published_paths: list[str] = []

    try:
        with TemporaryDirectory() as temporary_directory:
            downloaded_model = Path(
                model_downloader(
                    source_manifest.model.uri,
                    temporary_directory,
                )
            )

            sources = _release_sources(
                manifest=source_manifest,
                release_root=source_release_root,
            )
            sources.update(_model_sources(downloaded_model))

            published = publish_serving_artifacts(
                models_path=target_models_path,
                release_id=resolved_release_id,
                sources=sources,
            )
            published_paths = [reference.path for reference in published.references.values()]

        portable_model_uri = join_uri(
            published.release_root,
            "model",
        )
        metadata = dict(source_manifest.metadata or {})
        metadata["source_release_id"] = source_manifest.release_id
        metadata["source_model_uri"] = source_manifest.model.uri

        cloud_manifest = replace(
            source_manifest,
            release_id=resolved_release_id,
            created_at_utc=(datetime.now(UTC).isoformat()),
            model=replace(
                source_manifest.model,
                uri=portable_model_uri,
            ),
            artifacts=published.references,
            metadata=metadata,
        )

        write_serving_manifest(
            target_paths["manifest"],
            cloud_manifest,
        )

        persisted_manifest, release_root = load_release_manifest(
            models_path=target_models_path,
            release_id=resolved_release_id,
        )

        operation = (
            ReleaseOperation.ACTIVATION
            if file_exists(active_pointer_path(target_models_path))
            else ReleaseOperation.BOOTSTRAP
        )

        pointer = activate_release(
            models_path=target_models_path,
            release_id=resolved_release_id,
            operation=operation,
            expected_task_type=(source_manifest.task_type),
        )
    except Exception:
        _cleanup_export(
            target_models_path=(target_models_path),
            release_id=resolved_release_id,
            artifact_paths=published_paths,
        )
        raise

    return PublishedServingRelease(
        manifest=persisted_manifest,
        release_root=release_root,
        active_pointer=pointer,
    )
