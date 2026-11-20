"""Refresh ECS serving tasks and roll back model releases with registry alignment."""

import argparse
import json
import os
from typing import Any

import httpx
from mlflow import MlflowClient

from ..inference.bundle_loader import load_serving_bundle
from ..inference.model_loader import load_xgboost_model
from ..inference.releases.contracts import TaskType
from ..inference.releases.pointer import ReleaseOperation, load_active_release_pointer
from ..inference.releases.repository import activate_release, load_release_manifest
from .ecs_rollout import wait_for_rollout
from .lifecycle import load_aws_lifecycle_config


def refresh_service(*, expected_release_id: str, ecs: Any = None) -> None:
    """Restart tasks, wait for the deployment, and verify its loaded release."""
    cluster = os.environ["AWS_ECS_CLUSTER"]
    service = os.environ["AWS_API_SERVICE"]
    endpoint = os.environ["AWS_API_ENDPOINT"].rstrip("/")
    if ecs is None:
        import boto3

        ecs = boto3.client("ecs")
    response = ecs.describe_services(cluster=cluster, services=[service])
    services = response.get("services", [])
    if response.get("failures") or not services or services[0]["desiredCount"] < 1:
        raise RuntimeError("Serving requires an existing ECS service with running tasks.")
    deployment = ecs.update_service(cluster=cluster, service=service, forceNewDeployment=True)
    deployment_id = next(
        item["id"] for item in deployment["service"]["deployments"] if item["status"] == "PRIMARY"
    )
    ecs.get_waiter("services_stable").wait(
        cluster=cluster,
        services=[service],
        WaiterConfig={"Delay": 15, "MaxAttempts": 80},
    )
    wait_for_rollout(ecs=ecs, cluster=cluster, service=service, deployment_id=deployment_id)
    response = httpx.get(f"{endpoint}/readyz", timeout=30)
    response.raise_for_status()
    if response.json().get("active_release_id") != expected_release_id:
        raise RuntimeError("API readiness does not match the expected serving release.")


def rollback_model(*, config: dict[str, Any], refresh: bool = True) -> str:
    """Validate the previous bundle, align champion and pointer, then refresh ECS."""
    models_path = config["paths"]["models"]
    pointer = load_active_release_pointer(models_path=models_path)
    target = pointer.previous_release_id
    if target is None:
        raise ValueError("No previous release is available for rollback.")
    manifest, _ = load_release_manifest(models_path=models_path, release_id=target)
    if manifest.model.name != config["tracking"]["model_name"]:
        raise ValueError("Rollback release belongs to another registered model.")
    if not (manifest.metadata or {}).get("registry_registered"):
        raise ValueError("Rollback requires a release backed by a shared registry version.")
    load_serving_bundle(
        models_path=models_path,
        release_id=target,
        serving_alias="champion",
        model_loader=load_xgboost_model,
    )
    client = MlflowClient(tracking_uri=config["tracking"]["mlflow_tracking_uri"])
    client.get_model_version(name=manifest.model.name, version=manifest.model.version)
    old_champion = client.get_model_version_by_alias(name=manifest.model.name, alias="champion")
    client.set_registered_model_alias(
        name=manifest.model.name,
        alias="champion",
        version=manifest.model.version,
    )
    try:
        activate_release(
            models_path=models_path,
            release_id=target,
            operation=ReleaseOperation.ROLLBACK,
            previous_release_id=pointer.release_id,
            expected_task_type=TaskType.FORECASTING,
        )
    except Exception as activation_error:
        try:
            client.set_registered_model_alias(
                name=manifest.model.name,
                alias="champion",
                version=old_champion.version,
            )
        except Exception as restore_error:
            raise ExceptionGroup(
                "Rollback activation and alias restoration failed.",
                [activation_error, restore_error],
            ) from None
        raise
    if refresh:
        refresh_service(expected_release_id=target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["refresh", "rollback-model"])
    args = parser.parse_args()
    config = load_aws_lifecycle_config()
    if args.operation == "rollback-model":
        release_id = rollback_model(config=config)
    else:
        release_id = load_active_release_pointer(models_path=config["paths"]["models"]).release_id
        refresh_service(expected_release_id=release_id)
    print(json.dumps({"release_id": release_id, "operation": args.operation}))


if __name__ == "__main__":
    main()
