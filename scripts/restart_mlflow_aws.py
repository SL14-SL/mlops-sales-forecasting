"""Restart managed MLflow and verify that ECS completed the requested rollout."""

import json
import subprocess

import boto3

from mlops_sales_forecasting.aws.ecs_rollout import wait_for_rollout


def restart_service(*, ecs, cluster: str, service: str) -> None:
    before = ecs.list_tasks(cluster=cluster, serviceName=service, desiredStatus="RUNNING")[
        "taskArns"
    ]
    if not before:
        raise RuntimeError("MLflow persistence verification requires a running task.")
    result = ecs.update_service(cluster=cluster, service=service, forceNewDeployment=True)
    deployment_id = next(
        item["id"] for item in result["service"]["deployments"] if item["status"] == "PRIMARY"
    )
    ecs.get_waiter("services_stable").wait(
        cluster=cluster,
        services=[service],
        WaiterConfig={"Delay": 15, "MaxAttempts": 100},
    )
    wait_for_rollout(
        ecs=ecs,
        cluster=cluster,
        service=service,
        deployment_id=deployment_id,
        previous_task_arns=tuple(before),
    )


def main() -> None:
    raw = subprocess.check_output(
        [
            "terraform",
            "-chdir=infrastructure/terraform-aws",
            "output",
            "-json",
        ]
    )
    outputs = {key: value["value"] for key, value in json.loads(raw).items()}
    restart_service(
        ecs=boto3.client("ecs"),
        cluster=outputs["fargate_training_cluster_arn"],
        service=outputs["mlflow_service_name"],
    )
    print("MLflow ECS task replaced successfully.")


if __name__ == "__main__":
    main()
