from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.aws import lifecycle, operations


@pytest.mark.parametrize(
    "uri", ["", "sqlite:////tmp/job.db", "http://localhost:5000", "${MLFLOW_TRACKING_URI}"]
)
def test_requires_shared_registry(monkeypatch, uri):
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setattr(
        lifecycle, "load_config", lambda: {"tracking": {"mlflow_tracking_uri": uri}}
    )
    with pytest.raises(ValueError, match="shared HTTP"):
        lifecycle.load_aws_lifecycle_config()


def test_aws_config_uses_s3_and_full_training_parameters(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("S3_BUCKET_NAME", "offline-parity")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    config = lifecycle.load_aws_lifecycle_config()
    assert all(path.startswith("s3://offline-parity/") for path in config["paths"].values())
    assert config["model"]["params"]["n_estimators"] == 1200
    assert config["tracking"]["model_name"].endswith("prod")
    assert lifecycle.load_aws_lifecycle_config(estimators=3)["model"]["params"]["n_estimators"] == 3


def test_auto_retraining_refreshes_only_after_promotion(monkeypatch):
    config = {}
    monkeypatch.setattr(lifecycle, "load_aws_lifecycle_config", lambda **_: config)
    refresh = MagicMock()
    monkeypatch.setattr(lifecycle, "_refresh_promoted_release", refresh)
    for promoted in [False, True]:

        def retrain(*, after_lifecycle, config, promoted=promoted):
            after_lifecycle(SimpleNamespace(serving_release=object() if promoted else None))
            return SimpleNamespace(to_dict=lambda: {"champion_promoted": promoted})

        monkeypatch.setattr(lifecycle, "run_auto_retraining", retrain)
        assert lifecycle.run_aws_lifecycle(mode="auto-retrain")["champion_promoted"] == promoted
    refresh.assert_called_once_with(config)


def test_failed_ecs_refresh_rolls_back_then_reports_failure(monkeypatch):
    from mlops_sales_forecasting.inference.releases import pointer

    monkeypatch.setenv("AWS_API_SERVICE", "api")
    monkeypatch.setattr(
        pointer,
        "load_active_release_pointer",
        lambda **_: SimpleNamespace(release_id="new", previous_release_id="old"),
    )
    monkeypatch.setattr(
        operations, "refresh_service", MagicMock(side_effect=RuntimeError("unhealthy"))
    )
    rollback = MagicMock()
    monkeypatch.setattr(operations, "rollback_model", rollback)
    config = {"paths": {"models": "s3://bucket/models"}}
    with pytest.raises(RuntimeError, match="unhealthy"):
        lifecycle._refresh_promoted_release(config)
    rollback.assert_called_once_with(config=config)


def test_ecs_circuit_breaker_is_not_mistaken_for_success(monkeypatch):
    for name, value in {
        "AWS_ECS_CLUSTER": "cluster",
        "AWS_API_SERVICE": "api",
        "AWS_API_ENDPOINT": "https://api.example.com",
    }.items():
        monkeypatch.setenv(name, value)
    ecs = MagicMock()
    ecs.describe_services.side_effect = [
        {"services": [{"desiredCount": 1}]},
        {
            "services": [
                {"deployments": [{"id": "old", "status": "PRIMARY", "rolloutState": "COMPLETED"}]}
            ]
        },
    ]
    ecs.update_service.return_value = {
        "service": {"deployments": [{"id": "new", "status": "PRIMARY"}]}
    }
    with pytest.raises(RuntimeError, match="possible circuit rollback"):
        operations.refresh_service(expected_release_id="release-new", ecs=ecs)


def test_ecs_success_still_requires_expected_model(monkeypatch):
    for name, value in {
        "AWS_ECS_CLUSTER": "cluster",
        "AWS_API_SERVICE": "api",
        "AWS_API_ENDPOINT": "https://api.example.com",
    }.items():
        monkeypatch.setenv(name, value)
    ecs = MagicMock()
    ecs.describe_services.return_value = {
        "services": [
            {
                "desiredCount": 1,
                "deployments": [{"id": "new", "status": "PRIMARY", "rolloutState": "COMPLETED"}],
            }
        ]
    }
    ecs.update_service.return_value = {
        "service": {"deployments": [{"id": "new", "status": "PRIMARY"}]}
    }
    response = MagicMock()
    response.json.return_value = {"active_release_id": "wrong-release"}
    monkeypatch.setattr(operations.httpx, "get", lambda *_, **__: response)
    with pytest.raises(RuntimeError, match="expected serving release"):
        operations.refresh_service(expected_release_id="release-new", ecs=ecs)


def test_s3_lifecycle_selects_portable_publisher(monkeypatch):
    from mlops_sales_forecasting.orchestration import lifecycle_adapter

    pipeline = MagicMock()
    pipeline.config = {"paths": {"models": "s3://bucket/models"}}
    candidate = MagicMock()
    candidate.promotion.decision.promote = True
    monkeypatch.setattr(lifecycle_adapter, "build_serving_release_input", MagicMock())
    portable = MagicMock()
    registry_only = MagicMock()
    monkeypatch.setattr(lifecycle_adapter, "publish_portable_serving_release", portable)
    monkeypatch.setattr(lifecycle_adapter, "publish_serving_release", registry_only)
    lifecycle_adapter._publish_promoted_release(
        pipeline=pipeline,
        tracked_result=MagicMock(),
        candidate_result=candidate,
    )
    portable.assert_called_once()
    assert portable.call_args.kwargs["models_path"] == "s3://bucket/models"
    registry_only.assert_not_called()


def test_rollback_pointer_failure_restores_previous_champion(monkeypatch):
    monkeypatch.setattr(
        operations,
        "load_active_release_pointer",
        lambda **_: SimpleNamespace(release_id="new", previous_release_id="old"),
    )
    model = SimpleNamespace(name="test-model", version="1")
    manifest = SimpleNamespace(model=model, metadata={"registry_registered": True})
    monkeypatch.setattr(operations, "load_release_manifest", lambda **_: (manifest, "root"))
    monkeypatch.setattr(operations, "load_serving_bundle", MagicMock())
    client = MagicMock()
    client.get_model_version_by_alias.return_value.version = "2"
    monkeypatch.setattr(operations, "MlflowClient", lambda **_: client)
    monkeypatch.setattr(operations, "activate_release", MagicMock(side_effect=OSError("S3 failed")))
    config = {
        "paths": {"models": "s3://bucket/models"},
        "tracking": {
            "model_name": "test-model",
            "mlflow_tracking_uri": "https://mlflow.example.com",
        },
    }
    with pytest.raises(OSError, match="S3 failed"):
        operations.rollback_model(config=config, refresh=False)
    assert [
        call.kwargs["version"] for call in client.set_registered_model_alias.call_args_list
    ] == ["1", "2"]
