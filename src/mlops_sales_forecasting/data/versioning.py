from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import fsspec
import mlflow

from mlops_sales_forecasting.configs.paths import join_uri
from mlops_sales_forecasting.storage.filesystem import (
    ensure_dir,
    file_exists,
)
from mlops_sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)


def make_dataset_version() -> str:
    """Creates a UTC-based version id for one pipeline run."""
    return datetime.now(UTC).strftime("ds_%Y%m%d_%H%M%S")


def _join(
    base: str,
    *parts: str,
) -> str:
    """Join local paths and object-storage URIs."""
    return join_uri(
        base,
        *parts,
    )


def _require_path(
    config: Mapping[str, Any],
    name: str,
) -> str:
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a valid 'paths' section.")

    value = paths.get(name)

    if not isinstance(value, str) or not value:
        raise ValueError(f"Config must contain a non-empty 'paths.{name}' value.")

    return value


def _copy_file(
    src: str,
    dst: str,
) -> None:
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

    Priority:
    1. GIT_COMMIT_SHA environment variable, used in Docker/CI deployments.
    2. Local git command fallback, useful during local development.
    3. None if neither source is available.
    """
    env_commit = os.getenv("GIT_COMMIT_SHA")
    if env_commit:
        return env_commit.strip()

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


def get_active_config_name(
    config: Mapping[str, Any],
) -> str:
    """Return the active environment configuration name."""
    environment = str(
        config.get(
            "environment",
            "dev",
        )
    )
    return f"{environment}.yaml"


def build_snapshot_paths(
    version_id: str,
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Builds all versioned target paths for the current environment."""
    base = _require_path(
        config,
        "versioning",
    )

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


def snapshot_current_datasets(
    version_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Copies the current canonical datasets into a versioned snapshot structure.
    Works in both dev (local) and prod (GCS).
    """
    paths = build_snapshot_paths(
        version_id,
        config,
    )

    raw_path = _require_path(
        config,
        "raw_data",
    )
    validated_path = _require_path(
        config,
        "validated_data",
    )
    features_path = _require_path(
        config,
        "features",
    )
    splits_path = _require_path(
        config,
        "splits",
    )

    source_paths = {
        "raw_store": _join(
            raw_path,
            "store.csv",
        ),
        "raw_train": _join(
            raw_path,
            "train.csv",
        ),
        "raw_test": _join(
            raw_path,
            "test.csv",
        ),
        "validated_train": _join(
            validated_path,
            "train.parquet",
        ),
        "validated_store": _join(
            validated_path,
            "store.parquet",
        ),
        "features": _join(
            features_path,
            "features.parquet",
        ),
        "split_train": _join(
            splits_path,
            "train.parquet",
        ),
        "split_val": _join(
            splits_path,
            "val.parquet",
        ),
    }

    logger.info(f"📦 Creating dataset snapshot for version: {version_id}")

    for key, src in source_paths.items():
        _copy_file(src, paths[key])

    manifest = {
        "dataset_version": version_id,
        "environment": config.get("environment", "dev"),
        "config_name": get_active_config_name(config),
        "git_commit": get_git_commit(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": config.get("random_seed"),
        "effective_config": {
            "environment_config": dict(config),
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


def get_latest_dataset_manifest(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Loads the latest dataset manifest."""
    manifest_path = _join(
        _require_path(
            config,
            "versioning",
        ),
        "latest_manifest.json",
    )

    if not file_exists(manifest_path):
        raise FileNotFoundError(f"Latest manifest not found: {manifest_path}")

    with fsspec.open(manifest_path, "r") as f:
        return json.load(f)


def log_dataset_manifest_to_mlflow(
    manifest: Mapping[str, Any],
) -> None:
    """
    Log dataset lineage, snapshot paths and effective configuration to MLflow.

    Args:
        manifest: Dataset manifest produced by ``snapshot_current_datasets``.

    Notes:
        The function expects an active MLflow run.
    """
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


def get_dataset_paths_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, str | None]:
    """Extract versioned training paths from a dataset manifest."""
    snapshots = manifest["snapshots"]

    if not isinstance(snapshots, Mapping):
        raise ValueError("Dataset manifest must contain valid snapshots.")

    return {
        "train_file": snapshots["split_train"],
        "val_file": snapshots["split_val"],
        "features_file": snapshots.get("features"),
        "validated_train_file": snapshots.get("validated_train"),
        "validated_store_file": snapshots.get("validated_store"),
    }
