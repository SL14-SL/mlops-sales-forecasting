from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

@dataclass(frozen=True)
class ServingArtifactReference:
    """Reference to one immutable serving artifact."""

    path: str
    sha256: str


@dataclass(frozen=True)
class ServingReleaseManifest:
    """Persistent description of one complete serving release."""

    schema_version: int
    release_id: str
    created_at_utc: str

    model_name: str
    model_version: str
    model_run_id: str
    model_uri: str
    model_type: str
    target_transformation: str

    dataset_version: str | None
    config_hash: str | None
    git_commit: str | None

    store_metadata: ServingArtifactReference
    store_state: ServingArtifactReference
    known_calendar: ServingArtifactReference

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ServingBundle:
    """Complete validated state required for forecast serving."""

    release_id: str
    manifest: ServingReleaseManifest

    model: Any
    model_name: str
    model_type: str
    target_transformation: str
    serving_alias: str
    model_uri: str
    model_version: str
    model_run_id: str

    store_metadata: pd.DataFrame
    store_state: dict[str, Any]
    known_calendar: pd.DataFrame


def validate_serving_bundle(bundle: ServingBundle) -> None:
    """Raise ValueError if a serving bundle is incomplete or invalid."""

    if bundle.model is None:
        raise ValueError("Serving bundle has no model.")

    if not bundle.model_version:
        raise ValueError("Serving bundle has no model version.")

    if not bundle.model_run_id:
        raise ValueError("Serving bundle has no model run ID.")

    if bundle.store_metadata is None or bundle.store_metadata.empty:
        raise ValueError("Serving bundle has no store metadata.")

    if "Store" not in bundle.store_metadata.columns:
        raise ValueError(
            "Serving bundle store metadata is missing column 'Store'."
        )

    if bundle.store_state is None or not isinstance(
        bundle.store_state,
        dict,
    ):
        raise ValueError("Serving bundle has invalid forecasting state.")

    if bundle.known_calendar is None or bundle.known_calendar.empty:
        raise ValueError("Serving bundle has no known calendar.")

    if not bundle.release_id:
        raise ValueError(
            "Serving bundle has no release ID."
        )

    if bundle.manifest.release_id != bundle.release_id:
        raise ValueError(
            "Serving bundle release ID does not match manifest."
        )

    if bundle.manifest.model_name != bundle.model_name:
        raise ValueError(
            "Serving bundle model name does not match manifest."
        )

    if bundle.manifest.model_version != bundle.model_version:
        raise ValueError(
            "Serving bundle model version does not match manifest."
        )

    if bundle.manifest.model_run_id != bundle.model_run_id:
        raise ValueError(
            "Serving bundle model run ID does not match manifest."
        )