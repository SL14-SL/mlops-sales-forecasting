import json
import os
import socket
from pathlib import Path
from typing import Any

import gcsfs
import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.inference.router import load_registry_model
from src.utils.logger import get_logger

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
    except Exception as exc:
        logger.warning("Could not load store metadata: %s", exc)
        return None


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