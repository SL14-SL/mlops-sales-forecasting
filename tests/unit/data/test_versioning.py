import json
from unittest.mock import MagicMock

import pandas as pd

from src.data.versioning import (
    get_latest_dataset_manifest,
    log_dataset_manifest_to_mlflow,
    snapshot_current_datasets,
)


def test_snapshot_current_datasets_creates_manifest_and_files(tmp_path, monkeypatch):
    version_id = "ds_test_001"

    raw_dir = tmp_path / "data" / "raw"
    validated_dir = tmp_path / "data" / "validation"
    features_dir = tmp_path / "data" / "features"
    splits_dir = tmp_path / "data" / "splits"
    versioning_dir = tmp_path / "data" / "versioning"

    raw_dir.mkdir(parents=True, exist_ok=True)
    validated_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    versioning_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"Store": [1]}).to_csv(raw_dir / "store.csv", index=False)
    pd.DataFrame({"Store": [1], "Sales": [100]}).to_csv(raw_dir / "train.csv", index=False)
    pd.DataFrame({"Store": [1]}).to_csv(raw_dir / "test.csv", index=False)

    pd.DataFrame({"Store": [1], "Sales": [100]}).to_parquet(validated_dir / "train.parquet", index=False)
    pd.DataFrame({"Store": [1]}).to_parquet(validated_dir / "store.parquet", index=False)

    pd.DataFrame({"Store": [1], "feature_x": [0.5]}).to_parquet(features_dir / "features.parquet", index=False)

    pd.DataFrame({"Store": [1], "Sales": [100]}).to_parquet(splits_dir / "train.parquet", index=False)
    pd.DataFrame({"Store": [1], "Sales": [120]}).to_parquet(splits_dir / "val.parquet", index=False)

    def fake_get_path(name: str) -> str:
        mapping = {
            "raw_data": str(raw_dir),
            "validated_data": str(validated_dir),
            "features": str(features_dir),
            "splits": str(splits_dir),
            "versioning": str(versioning_dir),
        }
        return mapping[name]

    monkeypatch.setattr("src.data.versioning.get_path", fake_get_path)
    monkeypatch.setattr("src.data.versioning.get_git_commit", lambda: "abc123")
    monkeypatch.setattr("src.data.versioning.get_active_config_name", lambda: "dev.yaml")

    manifest = snapshot_current_datasets(version_id)

    snapshot_root = versioning_dir / version_id

    assert manifest["dataset_version"] == version_id
    assert manifest["git_commit"] == "abc123"
    assert manifest["config_name"] == "dev.yaml"
    assert (snapshot_root / "raw" / "store.csv").exists()
    assert (snapshot_root / "raw" / "train.csv").exists()
    assert (snapshot_root / "raw" / "test.csv").exists()
    assert (snapshot_root / "validated" / "train.parquet").exists()
    assert (snapshot_root / "validated" / "store.parquet").exists()
    assert (snapshot_root / "features" / "features.parquet").exists()
    assert (snapshot_root / "splits" / "train.parquet").exists()
    assert (snapshot_root / "splits" / "val.parquet").exists()
    assert (snapshot_root / "manifest.json").exists()
    assert (versioning_dir / "latest_manifest.json").exists()

    latest_manifest = json.loads((versioning_dir / "latest_manifest.json").read_text())
    assert latest_manifest["dataset_version"] == version_id
    assert latest_manifest["git_commit"] == "abc123"
    assert latest_manifest["config_name"] == "dev.yaml"


def test_get_latest_dataset_manifest_reads_latest_file(tmp_path, monkeypatch):
    versioning_dir = tmp_path / "data" / "versioning"
    versioning_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_version": "ds_test_002",
        "environment": "dev",
        "snapshots": {
            "features": "dummy/path/features.parquet"
        },
    }

    (versioning_dir / "latest_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    def fake_get_path(name: str) -> str:
        assert name == "versioning"
        return str(versioning_dir)

    monkeypatch.setattr("src.data.versioning.get_path", fake_get_path)

    manifest = get_latest_dataset_manifest()

    assert manifest["dataset_version"] == "ds_test_002"
    assert manifest["environment"] == "dev"


def test_log_dataset_manifest_to_mlflow_logs_params_and_artifact(monkeypatch):
    mock_log_param = MagicMock()
    mock_log_text = MagicMock()

    monkeypatch.setattr("src.data.versioning.mlflow.log_param", mock_log_param)
    monkeypatch.setattr("src.data.versioning.mlflow.log_text", mock_log_text)

    manifest = {
        "dataset_version": "ds_test_003",
        "environment": "dev",
        "config_name": "dev.yaml",
        "git_commit": "abc123",
        "snapshots": {
            "features": "/tmp/features.parquet",
            "split_train": "/tmp/train.parquet",
            "split_val": "/tmp/val.parquet",
        },
    }

    log_dataset_manifest_to_mlflow(manifest)

    mock_log_param.assert_any_call("dataset_version", "ds_test_003")
    mock_log_param.assert_any_call("dataset_environment", "dev")
    mock_log_param.assert_any_call("data_features_path", "/tmp/features.parquet")
    mock_log_param.assert_any_call("data_split_train_path", "/tmp/train.parquet")
    mock_log_param.assert_any_call("data_split_val_path", "/tmp/val.parquet")
    mock_log_param.assert_any_call("dataset_config_name", "dev.yaml")
    mock_log_param.assert_any_call("git_commit", "abc123")

    mock_log_text.assert_called_once()