import json
import os
import socket
import fsspec
from pathlib import Path
from typing import Any

import gcsfs
import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.inference.router import load_registry_model
from src.utils.logger import get_logger
from src.data.features.calendar import prepare_known_calendar_lookup
from src.inference.serving_bundle import ServingBundle, validate_serving_bundle

from src.inference.model_loader import (
    load_model_by_type,
)
from src.inference.serving_release import (
    load_active_serving_manifest,
    resolve_release_artifact_uri,
    load_serving_manifest,
)

logger = get_logger(__name__)


def resolve_tracking_uri(cfg: dict) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri is not None:
        return tracking_uri

    is_docker = os.path.exists("/.dockerenv")

    if is_docker:
        try:
            mlflow_ip = socket.gethostbyname("mlflow")
            return f"http://{mlflow_ip}:5000"
        except Exception:
            return "http://mlflow:5000"

    return cfg.get("mlflow_tracking_uri", "http://localhost:5000")


def load_store_metadata(
    *,
    validated_path: str,
    gcs_bucket: str | None,
) -> pd.DataFrame | None:
    if gcs_bucket and gcs_bucket != "None":
        store_file = f"gs://{gcs_bucket}/data/validation/store.parquet"
    else:
        store_file = f"{validated_path}/store.parquet"

    logger.info("Checking for store metadata at: %s", store_file)

    try:
        store_metadata = pd.read_parquet(store_file)
        store_metadata["Store"] = store_metadata["Store"].astype(int)
        logger.info("Store metadata loaded successfully.")
        return store_metadata
    except Exception:
        logger.exception(
            "Store metadata could not be loaded from %s.",
            store_file,
        )
        raise


def load_store_state(
    *,
    models_path: Path,
    gcs_bucket: str | None,
) -> dict[str, Any]:
    state_gcs_path = f"gs://{gcs_bucket}/models/latest_state.json"
    local_state_path = models_path / "latest_state.json"

    try:
        if gcs_bucket and gcs_bucket != "None":
            fs = gcsfs.GCSFileSystem()
            if fs.exists(state_gcs_path):
                with fs.open(state_gcs_path, "r") as f:
                    logger.info("Feature state loaded from GCS.")
                    return json.load(f)

            raise FileNotFoundError(f"State file not found on GCS: {state_gcs_path}")

        raise ValueError("No GCS bucket configured for state.")

    except Exception as exc:
        logger.warning("GCS state load failed: %s. Checking local fallback.", exc)

        if local_state_path.exists():
            with open(local_state_path, "r", encoding="utf-8") as f:
                logger.info("Feature state loaded from local path: %s", local_state_path)
                return json.load(f)

        logger.warning("No state snapshot found. Using empty state.")
        return {}

def load_known_calendar_artifact(
    *,
    features_path: str,
    gcs_bucket: str | None,
) -> pd.DataFrame:
    """Load the known calendar used during online inference."""
    if gcs_bucket and gcs_bucket != "None":
        calendar_path = (
            f"gs://{gcs_bucket}/data/features/"
            "known_calendar.parquet"
        )
    else:
        calendar_path = (
            f"{features_path}/known_calendar.parquet"
        )

    logger.info(
        "Checking for known calendar at: %s",
        calendar_path,
    )

    calendar_df = pd.read_parquet(calendar_path)

    calendar_df["Store"] = pd.to_numeric(
        calendar_df["Store"],
        errors="raise",
    ).astype(int)

    calendar_df["Date"] = pd.to_datetime(
        calendar_df["Date"],
        errors="raise",
    )

    logger.info(
        "Known calendar loaded successfully | rows=%s",
        len(calendar_df),
    )

    return prepare_known_calendar_lookup(
        calendar_df
    )

def reload_serving_model(
    *,
    model_name: str,
    cfg: dict,
) -> dict[str, Any]:
    """
    Reload the current forecasting champion model from MLflow Registry.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg))

    (
        model,
        model_type,
        target_transformation,
        serving_alias,
        model_uri,
    ) = load_registry_model(model_name)

    serving_model_version = None
    serving_model_run_id = None

    if serving_alias and serving_alias != "unknown":
        client = MlflowClient()
        version = client.get_model_version_by_alias(model_name, serving_alias)
        serving_model_version = str(version.version)
        serving_model_run_id = version.run_id
    else:
        raise RuntimeError(
            f"No valid serving alias resolved for model '{model_name}'."
        )

    logger.info(
        "Forecasting model reloaded: %s alias=%s version=%s run_id=%s",
        model_name,
        serving_alias,
        serving_model_version,
        serving_model_run_id,
    )

    return {
        "model": model,
        "model_type": model_type,
        "target_transformation": target_transformation,
        "serving_alias": serving_alias,
        "model_uri": model_uri,
        "serving_model_version": serving_model_version,
        "serving_model_run_id": serving_model_run_id,
        "model_name": model_name,
    }

def load_serving_bundle_for_release(
    *,
    release_id: str,
    model_name: str,
    cfg: dict,
    models_path: str | Path,
) -> ServingBundle:
    """
    Load and validate one concrete serving release without changing the
    active release pointer.
    """
    resolved_models_path = str(
        models_path
    )

    mlflow.set_tracking_uri(
        resolve_tracking_uri(cfg)
    )

    manifest, release_root = (
        load_serving_manifest(
            models_path=resolved_models_path,
            release_id=release_id,
        )
    )

    if manifest.model_name != model_name:
        raise ValueError(
            "Serving manifest model name does "
            "not match configuration: "
            f"{manifest.model_name} != "
            f"{model_name}"
        )

    metadata_uri = (
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=manifest.store_metadata,
        )
    )
    state_uri = (
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=manifest.store_state,
        )
    )
    calendar_uri = (
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=manifest.known_calendar,
        )
    )

    model = load_model_by_type(
        manifest.model_uri,
        manifest.model_type,
    )

    store_metadata = pd.read_parquet(
        metadata_uri
    )
    store_metadata["Store"] = pd.to_numeric(
        store_metadata["Store"],
        errors="raise",
    ).astype(int)

    with fsspec.open(
        state_uri,
        "r",
    ) as file_handle:
        store_state = json.load(
            file_handle
        )

    if not isinstance(store_state, dict):
        raise ValueError(
            "Serving release state must "
            "contain a JSON object."
        )

    known_calendar = pd.read_parquet(
        calendar_uri
    )
    known_calendar["Store"] = pd.to_numeric(
        known_calendar["Store"],
        errors="raise",
    ).astype(int)
    known_calendar["Date"] = pd.to_datetime(
        known_calendar["Date"],
        errors="raise",
    )

    known_calendar = (
        prepare_known_calendar_lookup(
            known_calendar
        )
    )

    bundle = ServingBundle(
        release_id=manifest.release_id,
        manifest=manifest,
        model=model,
        model_name=manifest.model_name,
        model_type=manifest.model_type,
        target_transformation=(
            manifest.target_transformation
        ),
        serving_alias="champion",
        model_uri=manifest.model_uri,
        model_version=manifest.model_version,
        model_run_id=manifest.model_run_id,
        store_metadata=store_metadata,
        store_state=store_state,
        known_calendar=known_calendar,
    )

    validate_serving_bundle(
        bundle
    )

    return bundle


def load_serving_bundle(
    *,
    model_name: str,
    cfg: dict,
    validated_path: str,
    features_path: str,
    models_path: str | Path,
    gcs_bucket: str | None,
) -> ServingBundle:
    """
    Load the currently active versioned serving release.
    """
    del validated_path
    del features_path
    del gcs_bucket

    resolved_models_path = str(
        models_path
    )

    manifest, _ = (
        load_active_serving_manifest(
            models_path=resolved_models_path,
        )
    )

    return load_serving_bundle_for_release(
        release_id=manifest.release_id,
        model_name=model_name,
        cfg=cfg,
        models_path=resolved_models_path,
    )