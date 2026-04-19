from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

import fsspec
import mlflow

from src.configs.loader import load_config, get_path, file_exists, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def make_dataset_version() -> str:
    """Creates a UTC-based version id for one pipeline run."""
    return datetime.now(timezone.utc).strftime("ds_%Y%m%d_%H%M%S")


def _join(base: str, *parts: str) -> str:
    """Joins local or gs:// paths safely."""
    if base.startswith("gs://"):
        return "/".join([base.rstrip("/"), *[p.strip("/") for p in parts]])
    return os.path.join(base, *parts)


def _copy_file(src: str, dst: str):
    """Copies one file locally or via fsspec."""
    if not src:
        return
    if not file_exists(src):
        logger.warning(f"Versioning skipped: Source not found -> {src}")
        return

    if not dst.startswith("gs://"):
        ensure_dir(os.path.dirname(dst))

    with fsspec.open(src, "rb") as fsrc, fsspec.open(dst, "wb") as fdst:
        fdst.write(fsrc.read())

def get_git_commit() -> str | None:
    """
    Returns the current git commit hash.
    If git is unavailable, returns None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not determine git commit: {e}")
        return None


def get_active_config_name() -> str:
    """
    Returns the active environment config name.
    """
    environment = CFG.get("environment", "dev")
    return f"{environment}.yaml"


def build_snapshot_paths(version_id: str) -> dict[str, str]:
    """Builds all versioned target paths for the current environment."""
    base = get_path("versioning")

    return {
        "base": _join(base, version_id),
        "raw_store": _join(base, version_id, "raw", "store.csv"),
        "raw_train": _join(base, version_id, "raw", "train.csv"),
        "raw_test": _join(base, version_id, "raw", "test.csv"),
        "validated_train": _join(base, version_id, "validated", "train.parquet"),
        "validated_store": _join(base, version_id, "validated", "store.parquet"),
        "features": _join(base, version_id, "features", "features.parquet"),
        "split_train": _join(base, version_id, "splits", "train.parquet"),
        "split_val": _join(base, version_id, "splits", "val.parquet"),
        "manifest": _join(base, version_id, "manifest.json"),
        "latest_manifest": _join(base, "latest_manifest.json"),
    }


def snapshot_current_datasets(version_id: str) -> dict:
    """
    Copies the current canonical datasets into a versioned snapshot structure.
    Works in both dev (local) and prod (GCS).
    """
    paths = build_snapshot_paths(version_id)

    source_paths = {
        "raw_store": _join(get_path("raw_data"), "store.csv"),
        "raw_train": _join(get_path("raw_data"), "train.csv"),
        "raw_test": _join(get_path("raw_data"), "test.csv"),
        "validated_train": _join(get_path("validated_data"), "train.parquet"),
        "validated_store": _join(get_path("validated_data"), "store.parquet"),
        "features": _join(get_path("features"), "features.parquet"),
        "split_train": _join(get_path("splits"), "train.parquet"),
        "split_val": _join(get_path("splits"), "val.parquet"),
    }

    logger.info(f"📦 Creating dataset snapshot for version: {version_id}")

    for key, src in source_paths.items():
        _copy_file(src, paths[key])

    manifest = {
        "dataset_version": version_id,
        "environment": CFG.get("environment", "dev"),
        "config_name": get_active_config_name(),
        "git_commit": get_git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": CFG.get("random_seed"),
        "effective_config": {
            "environment_config": CFG,
            "training_config": TRAIN_CFG,
        },
        "sources": source_paths,
        "snapshots": {
            "raw_store": paths["raw_store"],
            "raw_train": paths["raw_train"],
            "raw_test": paths["raw_test"],
            "validated_train": paths["validated_train"],
            "validated_store": paths["validated_store"],
            "features": paths["features"],
            "split_train": paths["split_train"],
            "split_val": paths["split_val"],
        },
    }

    if not paths["manifest"].startswith("gs://"):
        ensure_dir(os.path.dirname(paths["manifest"]))

    with fsspec.open(paths["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)

    with fsspec.open(paths["latest_manifest"], "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✅ Dataset manifest written to: {paths['manifest']}")
    return manifest


def get_latest_dataset_manifest() -> dict:
    """Loads the latest dataset manifest."""
    manifest_path = _join(get_path("versioning"), "latest_manifest.json")

    if not file_exists(manifest_path):
        raise FileNotFoundError(f"Latest manifest not found: {manifest_path}")

    with fsspec.open(manifest_path, "r") as f:
        return json.load(f)


def log_dataset_manifest_to_mlflow(manifest: dict):
    dataset_version = manifest["dataset_version"]

    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("dataset_environment", manifest.get("environment"))
    mlflow.log_param("dataset_config_name", manifest.get("config_name"))
    mlflow.log_param("git_commit", manifest.get("git_commit"))

    if manifest.get("seed") is not None:
        mlflow.log_param("seed", manifest.get("seed"))

    snapshots = manifest.get("snapshots", {})
    for key, value in snapshots.items():
        mlflow.log_param(f"data_{key}_path", value)

    manifest_json = json.dumps(manifest, indent=2)
    mlflow.log_text(manifest_json, f"dataset_manifest/{dataset_version}.json")

    effective_config = manifest.get("effective_config")
    if effective_config is not None:
        mlflow.log_text(
            json.dumps(effective_config, indent=2, sort_keys=True),
            f"dataset_manifest/{dataset_version}_effective_config.json",
        )

    logger.info(f"🧾 Logged dataset manifest to MLflow for version: {dataset_version}")

def get_dataset_paths_from_manifest(manifest: dict) -> dict[str,str]: 
    """
    Extracts the versioned training paths from a dataset manifest
    """
    snapshots = manifest["snapshots"]

    return {
        "train_file": snapshots["split_train"],
        "val_file": snapshots["split_val"],
        "features_file": snapshots.get("features"),
        "validated_train_file": snapshots.get("validated_train"),
        "validated_store_file": snapshots.get("validated_store"),
    }