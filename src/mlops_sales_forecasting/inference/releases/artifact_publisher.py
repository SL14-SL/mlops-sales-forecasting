from collections.abc import Mapping
from dataclasses import dataclass

from ...storage.filesystem import remove_file
from .contracts import ArtifactReference
from .storage import (
    build_artifact_reference,
    build_release_paths,
    copy_uri,
)


@dataclass(frozen=True)
class ServingArtifactSource:
    """Source and release-relative destination of an artifact."""

    source_uri: str
    relative_path: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_uri, str)
            or not self.source_uri.strip()
        ):
            raise ValueError(
                "Serving artifact source URI "
                "must not be empty."
            )

        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path.strip()
        ):
            raise ValueError(
                "Serving artifact relative path "
                "must not be empty."
            )


@dataclass(frozen=True)
class PublishedServingArtifacts:
    """References produced for one release directory."""

    release_root: str
    references: dict[
        str,
        ArtifactReference,
    ]


def _validate_sources(
    sources: Mapping[
        str,
        ServingArtifactSource,
    ],
) -> None:
    """Validate artifact names, values and destinations."""

    if not isinstance(sources, Mapping):
        raise TypeError(
            "Serving artifact sources must be a mapping."
        )

    if not sources:
        raise ValueError(
            "Serving release requires at least "
            "one artifact source."
        )

    relative_paths: list[str] = []

    for name, source in sources.items():
        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError(
                "Serving artifact name must not be empty."
            )

        if not isinstance(
            source,
            ServingArtifactSource,
        ):
            raise TypeError(
                "Serving artifact source must be "
                "ServingArtifactSource."
            )

        relative_paths.append(
            source.relative_path
        )

    if len(relative_paths) != len(
        set(relative_paths)
    ):
        raise ValueError(
            "Serving artifact destinations "
            "must be unique."
        )


def _remove_copied_artifacts(
    copied_paths: list[str],
) -> None:
    """Best-effort cleanup of partially copied artifacts."""

    for copied_path in reversed(
        copied_paths
    ):
        try:
            remove_file(copied_path)
        except Exception:
            continue


def publish_serving_artifacts(
    *,
    models_path: str,
    release_id: str,
    sources: Mapping[
        str,
        ServingArtifactSource,
    ],
) -> PublishedServingArtifacts:
    """Copy and checksum all artifacts for one release."""

    _validate_sources(sources)

    artifact_filenames = {
        name: source.relative_path
        for name, source in sources.items()
    }
    release_paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
        artifact_filenames=artifact_filenames,
    )

    copied_paths: list[str] = []
    references: dict[
        str,
        ArtifactReference,
    ] = {}

    try:
        for name, source in sources.items():
            target_uri = release_paths[name]

            if source.source_uri == target_uri:
                raise ValueError(
                    "Serving artifact source and "
                    "destination must differ."
                )

            copy_uri(
                source.source_uri,
                target_uri,
            )
            copied_paths.append(target_uri)

            references[name] = (
                build_artifact_reference(
                    relative_path=(
                        source.relative_path
                    ),
                    artifact_uri=target_uri,
                )
            )
    except Exception:
        _remove_copied_artifacts(
            copied_paths
        )
        raise

    return PublishedServingArtifacts(
        release_root=(
            release_paths["release_root"]
        ),
        references=references,
    )