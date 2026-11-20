"""Publish a registered champion with all model files in object storage."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import mlflow

from ...configs.paths import join_uri
from ...tracking.promotion_service import PromotionOutcome
from ...tracking.registry import ModelRegistrationResult
from ..model_loader import load_xgboost_model
from .artifact_publisher import ServingArtifactSource
from .cloud_exporter import _model_sources
from .contracts import TaskType
from .publisher import PublishedServingRelease, publish_serving_release
from .storage import build_release_paths


def publish_portable_serving_release(
    *,
    models_path: str,
    registration: ModelRegistrationResult,
    promotion: PromotionOutcome,
    task_type: TaskType,
    model_type: str,
    sources: Mapping[str, ServingArtifactSource],
    metadata: Mapping[str, Any] | None = None,
    dataset_version: str | None = None,
    config_hash: str | None = None,
    git_commit: str | None = None,
) -> PublishedServingRelease:
    """Validate and copy a model before the shared publisher activates its release."""
    if not registration.registered or not registration.model_uri:
        raise ValueError("Portable publication requires a registered model.")
    if not promotion.decision.promote:
        raise ValueError("Portable publication requires a promoted champion.")
    if model_type != "xgboost":
        raise ValueError("Portable forecasting publication requires XGBoost.")

    release_id = f"release-{uuid4()}"
    paths = build_release_paths(models_path=models_path, release_id=release_id)
    with TemporaryDirectory(prefix="forecasting-release-") as temporary:
        model_root = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=registration.model_uri,
                dst_path=temporary,
            )
        )
        # Check the signature and deserialize the complete model before activation.
        load_xgboost_model(str(model_root))
        model_sources = _model_sources(model_root)
        if set(sources).intersection(model_sources):
            raise ValueError("Serving sources contain reserved model file names.")
        portable_uri = join_uri(paths["release_root"], "model")
        return publish_serving_release(
            models_path=models_path,
            registration=replace(registration, model_uri=portable_uri),
            promotion=promotion,
            task_type=task_type,
            model_type=model_type,
            sources={**sources, **model_sources},
            metadata={
                **(metadata or {}),
                "source_model_uri": registration.model_uri,
                "registry_registered": True,
            },
            dataset_version=dataset_version,
            config_hash=config_hash,
            git_commit=git_commit,
            release_id=release_id,
            validate_bundle=True,
        )
