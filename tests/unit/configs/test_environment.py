import os

import pytest

from mlops_sales_forecasting.configs.environment import (
    detect_environment,
    inject_runtime_env,
    override_gcs_bucket_paths,
    resolve_env_placeholders,
)


def test_detect_environment_uses_app_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "Staging")
    monkeypatch.setenv("K_SERVICE", "cloud-run-service")

    assert detect_environment() == "staging"


def test_detect_environment_detects_cloud_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("K_SERVICE", "cloud-run-service")

    assert detect_environment() == "prod"


def test_detect_environment_defaults_to_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)

    assert detect_environment() == "dev"


def test_resolve_env_placeholders_uses_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_HOST", "production-host")

    result = resolve_env_placeholders("https://${EXAMPLE_HOST}/api")

    assert result == "https://production-host/api"


def test_resolve_env_placeholders_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXAMPLE_PORT", raising=False)

    result = resolve_env_placeholders("${EXAMPLE_PORT:-8080}")

    assert result == "8080"


def test_resolve_env_placeholders_preserves_unresolved_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_VARIABLE", raising=False)

    result = resolve_env_placeholders("${MISSING_VARIABLE}")

    assert result == "${MISSING_VARIABLE}"


def test_resolve_env_placeholders_handles_nested_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_BUCKET", "customer-bucket")

    config = {
        "paths": {
            "raw": "gs://${EXAMPLE_BUCKET}/data/raw",
        },
        "services": [
            "${SERVICE_NAME:-api}",
            42,
        ],
    }

    result = resolve_env_placeholders(config)

    assert result == {
        "paths": {
            "raw": "gs://customer-bucket/data/raw",
        },
        "services": [
            "api",
            42,
        ],
    }


def test_resolve_env_placeholders_preserves_other_types() -> None:
    assert resolve_env_placeholders(42) == 42
    assert resolve_env_placeholders(True) is True
    assert resolve_env_placeholders(None) is None


def test_override_gcs_bucket_paths_replaces_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCS_BUCKET_NAME", "new-bucket")

    config = {
        "paths": {
            "raw": "gs://old-bucket/data/raw",
            "processed": "gs://old-bucket/data/processed",
            "local": "data/local",
        }
    }

    result = override_gcs_bucket_paths(config)

    assert result["paths"] == {
        "raw": "gs://new-bucket/data/raw",
        "processed": "gs://new-bucket/data/processed",
        "local": "data/local",
    }


def test_override_gcs_bucket_paths_normalizes_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GCS_BUCKET_NAME",
        "gs://new-bucket/",
    )

    config = {
        "paths": {
            "raw": "gs://old-bucket/data/raw",
        }
    }

    result = override_gcs_bucket_paths(config)

    assert result["paths"]["raw"] == "gs://new-bucket/data/raw"


def test_override_gcs_bucket_paths_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)

    config = {
        "paths": {
            "raw": "gs://original-bucket/data/raw",
        }
    }

    result = override_gcs_bucket_paths(config)

    assert result == config


def test_override_gcs_bucket_paths_without_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCS_BUCKET_NAME", "new-bucket")

    config = {"project": {"name": "example"}}

    assert override_gcs_bucket_paths(config) == config


def test_inject_runtime_env_sets_service_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = {
        "services": {
            "prefect_api_url": "http://prefect:4200/api",
        },
        "tracking": {
            "mlflow_tracking_uri": "http://mlflow:5000",
        },
    }

    inject_runtime_env(config)

    assert os.environ["PREFECT_API_URL"] == "http://prefect:4200/api"
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_inject_runtime_env_supports_legacy_mlflow_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = {
        "mlflow_tracking_uri": "http://localhost:5000",
    }

    inject_runtime_env(config)

    assert os.environ["MLFLOW_TRACKING_URI"] == "http://localhost:5000"


def test_inject_runtime_env_does_not_override_existing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PREFECT_API_URL",
        "http://existing-prefect:4200/api",
    )
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://existing-mlflow:5000",
    )

    config = {
        "services": {
            "prefect_api_url": "http://new-prefect:4200/api",
        },
        "tracking": {
            "mlflow_tracking_uri": "http://new-mlflow:5000",
        },
    }

    inject_runtime_env(config)

    assert os.environ["PREFECT_API_URL"] == "http://existing-prefect:4200/api"
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://existing-mlflow:5000"


def test_override_gcs_bucket_paths_maps_local_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GCS_BUCKET_NAME",
        "forecasting-dev-artifacts",
    )

    config = {
        "paths": {
            "data_root": "data",
            "raw_data": "data/raw",
            "artifacts": "artifacts",
            "models": "artifacts/models",
            "monitoring": "data/monitoring",
            "predictions": "data/predictions",
        }
    }

    result = override_gcs_bucket_paths(config)

    assert result["paths"] == {
        "data_root": ("gs://forecasting-dev-artifacts/data"),
        "raw_data": ("gs://forecasting-dev-artifacts/data/raw"),
        "artifacts": ("gs://forecasting-dev-artifacts/artifacts"),
        "models": ("gs://forecasting-dev-artifacts/models"),
        "monitoring": ("gs://forecasting-dev-artifacts/data/monitoring"),
        "predictions": ("gs://forecasting-dev-artifacts/data/predictions"),
    }
