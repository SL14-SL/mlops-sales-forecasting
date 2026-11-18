import importlib.util
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.configs.paths import get_project_root
from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)


@pytest.fixture
def verifier():
    path = get_project_root() / "scripts/verify_aws_model_rollout.py"
    spec = importlib.util.spec_from_file_location("rollout_verifier_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifest():
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-original",
        created_at_utc="2026-11-18T12:00:00+00:00",
        task_type=TaskType.FORECASTING,
        model=ModelReference(
            name="model-dev",
            version="1",
            run_id="training-run",
            uri="s3://bucket/models/serving_releases/release-original/model",
            model_type="xgboost",
        ),
        artifacts={"model": ArtifactReference(path="model/model.ubj", sha256="a" * 64)},
        metadata={"registry_registered": True},
    )


def test_logged_model_id_is_passed_even_when_version_field_is_missing(
    verifier, manifest, monkeypatch
):
    client = MagicMock()
    client.get_model_version.return_value = SimpleNamespace(
        source="models:/m-logged-model", run_id="training-run", model_id=None
    )
    client.create_model_version.return_value = SimpleNamespace(status="READY", version="2")
    monkeypatch.setattr(verifier, "load_serving_bundle", MagicMock())
    copy = MagicMock()
    monkeypatch.setattr(verifier, "copy_uri", copy)
    monkeypatch.setattr(verifier, "write_serving_manifest", MagicMock())

    duplicate = verifier.prepare_duplicate(
        models_path="s3://bucket/models",
        manifest=manifest,
        release_root="s3://bucket/models/serving_releases/release-original",
        client=client,
    )

    assert client.create_model_version.call_args.kwargs["model_id"] == "m-logged-model"
    assert duplicate.release_id != manifest.release_id
    assert duplicate.artifacts == manifest.artifacts
    assert duplicate.metadata["identical_weights"] is True
    client.set_registered_model_alias.assert_not_called()
    assert copy.call_args.args[0] != copy.call_args.args[1]


def test_failed_duplicate_validation_does_not_activate_or_change_alias(
    verifier, manifest, monkeypatch
):
    client = MagicMock()
    client.get_model_version.return_value = SimpleNamespace(
        source="s3://bucket/model", run_id="run"
    )
    client.create_model_version.return_value = SimpleNamespace(status="READY", version="2")
    monkeypatch.setattr(
        verifier,
        "load_serving_bundle",
        MagicMock(side_effect=[object(), ValueError("checksum mismatch")]),
    )
    monkeypatch.setattr(verifier, "copy_uri", MagicMock())
    monkeypatch.setattr(verifier, "write_serving_manifest", MagicMock())
    activate = MagicMock()
    monkeypatch.setattr(verifier, "activate_release", activate)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verifier.prepare_duplicate(
            models_path="s3://bucket/models",
            manifest=manifest,
            release_root="s3://bucket/models/serving_releases/release-original",
            client=client,
        )
    activate.assert_not_called()
    client.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("changed", ["pointer", "champion"])
def test_concurrent_change_blocks_cutover(verifier, manifest, monkeypatch, tmp_path, changed):
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="9" if changed == "champion" else "1"
    )
    monkeypatch.setattr(
        verifier,
        "load_active_release_pointer",
        lambda **kwargs: SimpleNamespace(
            release_id="release-other" if changed == "pointer" else manifest.release_id
        ),
    )
    activate = MagicMock()
    rollback = MagicMock()
    monkeypatch.setattr(verifier, "activate_release", activate)
    monkeypatch.setattr(verifier, "rollback_model", rollback)
    with pytest.raises(RuntimeError, match="changed during preparation"):
        verifier.verify_switch_and_rollback(
            config={"paths": {"models": "s3://bucket/models"}},
            client=client,
            original=manifest,
            duplicate=replace(manifest, release_id="release-test"),
            observe=MagicMock(),
            evidence=tmp_path,
        )
    client.set_registered_model_alias.assert_not_called()
    activate.assert_not_called()
    rollback.assert_not_called()


def test_wrong_expected_release_stops_before_creating_a_test_version(verifier, monkeypatch):
    import json
    import sys

    outputs = {
        name: {"value": value}
        for name, value in {
            "aws_region": "eu-north-1",
            "fargate_training_cluster_arn": "cluster",
            "api_service_name": "api",
            "api_endpoint": "https://api.example.com",
            "mlflow_tracking_uri": "https://mlflow.example.com",
            "mlflow_token_secret_arn": "mlflow-secret",
            "api_key_secret_arn": "api-secret",
            "serving_models_uri": "s3://bucket/models",
        }.items()
    }
    monkeypatch.setattr(verifier.subprocess, "check_output", lambda *args: json.dumps(outputs))
    secrets = MagicMock()
    secrets.get_secret_value.return_value = {"SecretString": "test-secret"}
    monkeypatch.setattr(verifier.boto3, "client", lambda *args, **kwargs: secrets)
    client = MagicMock()
    monkeypatch.setattr(verifier, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(
        verifier,
        "load_active_release_pointer",
        lambda **kwargs: SimpleNamespace(release_id="release-actual"),
    )
    monkeypatch.setattr(sys, "argv", ["verify", "--expected-release", "release-stale"])
    # main sets runtime environment variables; keep them inside this test's scope.
    for name in (
        "AWS_DEFAULT_REGION",
        "AWS_ECS_CLUSTER",
        "AWS_API_SERVICE",
        "AWS_API_ENDPOINT",
        "MLFLOW_TRACKING_TOKEN",
    ):
        monkeypatch.setenv(name, "before-test")
    with pytest.raises(RuntimeError, match="expected starting release"):
        verifier.main()
    client.create_model_version.assert_not_called()
    client.set_registered_model_alias.assert_not_called()
