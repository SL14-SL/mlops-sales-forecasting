from collections.abc import Callable
from typing import Any

import pandas as pd

from .releases.contracts import (
    ArtifactReference,
    ServingReleaseManifest,
)
from .releases.policy import validate_task_manifest
from .releases.repository import (
    load_active_release_manifest,
    load_release_manifest,
)
from .releases.storage import (
    load_json,
    resolve_artifact_uri,
)
from .serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)

ModelLoader = Callable[[str], Any]


def _required_artifact(
    manifest: ServingReleaseManifest,
    name: str,
) -> ArtifactReference:
    """Return a required artifact reference."""
    reference = manifest.artifacts.get(name)

    if reference is None:
        raise ValueError(
            f"Serving manifest is missing required artifact: {name}."
        )

    return reference


def build_serving_bundle(
    *,
    manifest: ServingReleaseManifest,
    release_root: str,
    serving_alias: str,
    model_loader: ModelLoader,
) -> ServingBundle:
    """Load and validate a complete serving bundle."""
    validate_task_manifest(manifest)
    model = model_loader(manifest.model.uri)


    store_metadata_uri = resolve_artifact_uri(
        release_root=release_root,
        reference=_required_artifact(
            manifest,
            "store_metadata",
        ),
    )
    store_state_uri = resolve_artifact_uri(
        release_root=release_root,
        reference=_required_artifact(
            manifest,
            "store_state",
        ),
    )
    known_calendar_uri = resolve_artifact_uri(
        release_root=release_root,
        reference=_required_artifact(
            manifest,
            "known_calendar",
        ),
    )
    target_transformation = (
        manifest.metadata or {}
    )["target_transformation"]

    bundle = ServingBundle(
        release_id=manifest.release_id,
        manifest=manifest,
        model=model,
        serving_alias=serving_alias,
        target_transformation=str(target_transformation),
        store_metadata=pd.read_parquet(store_metadata_uri),
        store_state=load_json(store_state_uri),
        known_calendar=pd.read_parquet(known_calendar_uri),
    )


    validate_serving_bundle(bundle)
    return bundle


def load_serving_bundle(
    *,
    models_path: str,
    release_id: str,
    serving_alias: str,
    model_loader: ModelLoader,
) -> ServingBundle:
    """Load a serving bundle for a specific release."""
    manifest, release_root = load_release_manifest(
        models_path=models_path,
        release_id=release_id,
    )

    return build_serving_bundle(
        manifest=manifest,
        release_root=release_root,
        serving_alias=serving_alias,
        model_loader=model_loader,
    )


def load_active_serving_bundle(
    *,
    models_path: str,
    serving_alias: str,
    model_loader: ModelLoader,
) -> ServingBundle:
    """Load the currently active serving bundle."""
    manifest, release_root = load_active_release_manifest(
        models_path=models_path
    )

    return build_serving_bundle(
        manifest=manifest,
        release_root=release_root,
        serving_alias=serving_alias,
        model_loader=model_loader,
    )