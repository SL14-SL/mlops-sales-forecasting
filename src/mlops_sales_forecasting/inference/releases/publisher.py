from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ...storage.filesystem import (
    file_exists,
    remove_file,
)
from ...tracking.promotion_service import (
    PromotionOutcome,
)
from ...tracking.registry import (
    ModelRegistrationResult,
)
from .artifact_publisher import (
    PublishedServingArtifacts,
    ServingArtifactSource,
    publish_serving_artifacts,
)
from .builder import (
    build_serving_release_manifest,
)
from .contracts import (
    ArtifactReference,
    ServingReleaseManifest,
    TaskType,
)
from .manifest import write_serving_manifest
from .pointer import (
    ActiveReleasePointer,
    ReleaseOperation,
    load_active_release_pointer,
)
from .repository import (
    activate_release,
    load_release_manifest,
)
from .storage import (
    build_artifact_reference,
    build_release_paths,
)


@dataclass(frozen=True)
class PublishedServingRelease:
    """Fully persisted and activated serving release."""

    manifest: ServingReleaseManifest
    release_root: str
    active_pointer: ActiveReleasePointer


def _new_release_id() -> str:
    """Return a unique release identifier."""

    return f"release-{uuid4()}"


def _build_source_references(
    sources: Mapping[
        str,
        ServingArtifactSource,
    ],
) -> dict[str, ArtifactReference]:
    """Build references from sources before copying."""

    return {
        name: build_artifact_reference(
            relative_path=(
                source.relative_path
            ),
            artifact_uri=source.source_uri,
        )
        for name, source in sources.items()
    }


def _cleanup_release(
    *,
    models_path: str,
    release_id: str,
    references: Mapping[
        str,
        ArtifactReference,
    ],
) -> None:
    """Best-effort removal of an incomplete release."""

    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
        artifact_filenames={
            name: reference.path
            for name, reference
            in references.items()
        },
    )

    cleanup_paths = [
        paths[name]
        for name in references
    ]
    cleanup_paths.append(
        paths["manifest"]
    )

    for path in reversed(cleanup_paths):
        try:
            remove_file(path)
        except Exception:
            continue


def _previous_release(
    *,
    models_path: str,
    release_id: str,
) -> tuple[
    ReleaseOperation,
    str | None,
]:
    """Return activation type and previous release ID."""

    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if not file_exists(
        paths["active_pointer"]
    ):
        return (
            ReleaseOperation.BOOTSTRAP,
            None,
        )

    current_pointer = (
        load_active_release_pointer(
            models_path=models_path
        )
    )

    return (
        ReleaseOperation.ACTIVATION,
        current_pointer.release_id,
    )


def publish_serving_release(
    *,
    models_path: str,
    registration: ModelRegistrationResult,
    promotion: PromotionOutcome,
    task_type: TaskType,
    model_type: str,
    sources: Mapping[
        str,
        ServingArtifactSource,
    ],
    metadata: Mapping[str, Any] | None = None,
    dataset_version: str | None = None,
    config_hash: str | None = None,
    git_commit: str | None = None,
    release_id: str | None = None,
) -> PublishedServingRelease:
    """Validate, persist and activate one serving release."""

    resolved_release_id = (
        release_id or _new_release_id()
    )

    source_references = (
        _build_source_references(
            sources
        )
    )

    manifest = build_serving_release_manifest(
        registration=registration,
        promotion=promotion,
        task_type=task_type,
        model_type=model_type,
        artifacts=source_references,
        metadata=metadata,
        dataset_version=dataset_version,
        config_hash=config_hash,
        git_commit=git_commit,
        release_id=resolved_release_id,
    )

    published_artifacts: (
        PublishedServingArtifacts | None
    ) = None

    try:
        published_artifacts = (
            publish_serving_artifacts(
                models_path=models_path,
                release_id=(
                    resolved_release_id
                ),
                sources=sources,
            )
        )

        if (
            published_artifacts.references
            != source_references
        ):
            raise ValueError(
                "Serving artifact contents changed "
                "during release publication."
            )

        paths = build_release_paths(
            models_path=models_path,
            release_id=(
                resolved_release_id
            ),
        )
        write_serving_manifest(
            paths["manifest"],
            manifest,
        )

        persisted_manifest, release_root = (
            load_release_manifest(
                models_path=models_path,
                release_id=(
                    resolved_release_id
                ),
            )
        )
    except Exception:
        references = (
            published_artifacts.references
            if published_artifacts is not None
            else source_references
        )
        _cleanup_release(
            models_path=models_path,
            release_id=resolved_release_id,
            references=references,
        )
        raise

    operation, previous_release_id = (
        _previous_release(
            models_path=models_path,
            release_id=resolved_release_id,
        )
    )
    pointer = activate_release(
        models_path=models_path,
        release_id=resolved_release_id,
        operation=operation,
        expected_task_type=task_type,
        previous_release_id=(
            previous_release_id
        ),
    )

    return PublishedServingRelease(
        manifest=persisted_manifest,
        release_root=release_root,
        active_pointer=pointer,
    )