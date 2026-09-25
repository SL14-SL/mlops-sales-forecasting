from dataclasses import dataclass
from typing import Any

import pandas as pd

from .releases.contracts import (
    ServingReleaseManifest,
    TaskType,
)
from .releases.policy import validate_task_manifest


@dataclass(frozen=True)
class ServingBundle:
    """Complete validated state required for model inference."""

    release_id: str
    manifest: ServingReleaseManifest
    model: Any
    serving_alias: str


    target_transformation: str
    store_metadata: pd.DataFrame
    store_state: dict[str, Any]
    known_calendar: pd.DataFrame


    @property
    def model_name(self) -> str:
        """Return the registered model name."""
        return self.manifest.model.name

    @property
    def model_version(self) -> str:
        """Return the registered model version."""
        return self.manifest.model.version

    @property
    def model_run_id(self) -> str:
        """Return the MLflow run identifier."""
        return self.manifest.model.run_id

    @property
    def model_uri(self) -> str:
        """Return the registered model URI."""
        return self.manifest.model.uri

    @property
    def model_type(self) -> str:
        """Return the model implementation type."""
        return self.manifest.model.model_type


def _validate_common_bundle(bundle: ServingBundle) -> None:
    """Validate fields shared by all serving bundles."""
    if not isinstance(bundle, ServingBundle):
        raise ValueError("Invalid serving bundle.")

    validate_task_manifest(bundle.manifest)

    if not bundle.release_id:
        raise ValueError(
            "Serving bundle has no release ID."
        )

    if bundle.manifest.release_id != bundle.release_id:
        raise ValueError(
            "Serving bundle release ID does not match manifest."
        )

    if bundle.model is None:
        raise ValueError(
            "Serving bundle has no model."
        )

    if not bundle.serving_alias:
        raise ValueError(
            "Serving bundle has no serving alias."
        )


def validate_serving_bundle(bundle: ServingBundle) -> None:
    """Validate a complete task-specific serving bundle."""
    _validate_common_bundle(bundle)


    if bundle.manifest.task_type is not TaskType.FORECASTING:
        raise ValueError(
            "Serving bundle requires a forecasting manifest."
        )

    manifest_transformation = (
        bundle.manifest.metadata or {}
    ).get("target_transformation")

    if (
        not bundle.target_transformation
        or bundle.target_transformation
        != manifest_transformation
    ):
        raise ValueError(
            "Serving bundle target transformation does not "
            "match manifest."
        )

    if (
        not isinstance(bundle.store_metadata, pd.DataFrame)
        or bundle.store_metadata.empty
    ):
        raise ValueError(
            "Serving bundle has no store metadata."
        )

    if (
        not isinstance(bundle.store_state, dict)
        or not bundle.store_state
    ):
        raise ValueError(
            "Serving bundle has invalid forecasting state."
        )

    if (
        not isinstance(bundle.known_calendar, pd.DataFrame)
        or bundle.known_calendar.empty
    ):
        raise ValueError(
            "Serving bundle has no known calendar."
        )
