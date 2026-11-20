from pathlib import Path

import pytest

from mlops_sales_forecasting.configs import loader
from mlops_sales_forecasting.configs.environment import override_storage_paths


@pytest.fixture(autouse=True)
def clean_cloud_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLOUD_PROVIDER",
        "S3_BUCKET_NAME",
        "GCS_BUCKET_NAME",
        "K_SERVICE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
def test_loads_aws_paths_for_each_environment(
    environment: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    (config_directory / f"{environment}.yaml").write_text(
        "paths:\n  raw_data: data/raw\n  models: artifacts/models\n  monitoring: data/monitoring\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("APP_ENV", environment)
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("S3_BUCKET_NAME", "forecasting-test")
    monkeypatch.setenv("GCS_BUCKET_NAME", "old-gcp-bucket")

    config = loader.load_config()

    assert config["environment"] == environment
    assert config["paths"] == {
        "raw_data": "s3://forecasting-test/data/raw",
        "models": "s3://forecasting-test/models",
        "monitoring": "s3://forecasting-test/data/monitoring",
    }


def test_aws_requires_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")

    with pytest.raises(ValueError, match="Set S3_BUCKET_NAME"):
        override_storage_paths({"paths": {"models": "artifacts/models"}})


@pytest.mark.parametrize(
    "bucket",
    ["s3://", "bucket/prefix", "gs://bucket", "bad bucket", "${BUCKET}"],
)
def test_aws_rejects_invalid_bucket_values(
    bucket: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("S3_BUCKET_NAME", bucket)

    with pytest.raises(ValueError, match="without a prefix"):
        override_storage_paths({"paths": {"models": "artifacts/models"}})


def test_aws_rejects_unmapped_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("S3_BUCKET_NAME", "forecasting-test")

    with pytest.raises(ValueError, match="No AWS path mapping"):
        override_storage_paths({"paths": {"unknown": "data/unknown"}})


def test_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_PROVIDER", "awss")

    with pytest.raises(ValueError, match="CLOUD_PROVIDER"):
        override_storage_paths({"paths": {"models": "artifacts/models"}})


def test_preserves_legacy_gcs_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCS_BUCKET_NAME", "existing-gcp-bucket")

    config = override_storage_paths({"paths": {"models": "artifacts/models"}})

    assert config["paths"]["models"] == "gs://existing-gcp-bucket/models"
