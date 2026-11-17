"""Run a lifecycle task from Terraform outputs and propagate container failures."""

import argparse
import json
import subprocess
import time


def aws(*arguments: str) -> dict:
    result = subprocess.run(
        ["aws", *arguments, "--output", "json"], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "auto-retrain", "rollback-model", "refresh"])
    parser.add_argument("--estimators", type=int)
    args = parser.parse_args()
    output = subprocess.run(
        ["terraform", "-chdir=infrastructure/terraform-aws", "output", "-json"],
        capture_output=True,
        text=True,
        check=True,
    )
    values = {name: value["value"] for name, value in json.loads(output.stdout).items()}
    command = [args.mode]
    if args.estimators is not None:
        if args.estimators < 1:
            parser.error("estimators must be positive")
        command += ["--estimators", str(args.estimators)]
    response = aws(
        "ecs",
        "run-task",
        "--cluster",
        values["fargate_training_cluster_arn"],
        "--task-definition",
        values["fargate_training_task_definition_arn"],
        "--launch-type",
        "FARGATE",
        "--network-configuration",
        json.dumps(
            {
                "awsvpcConfiguration": {
                    "subnets": [values["training_subnet_id"]],
                    "securityGroups": [values["training_security_group_id"]],
                    "assignPublicIp": "ENABLED",
                },
            }
        ),
        "--overrides",
        json.dumps({"containerOverrides": [{"name": "training", "command": command}]}),
    )
    if response.get("failures") or not response.get("tasks"):
        raise RuntimeError(f"ECS could not launch the task: {response.get('failures')}")
    task_arn = response["tasks"][0]["taskArn"]
    print(f"Lifecycle task: {task_arn}", flush=True)
    # Training plus model validation and serving refresh may take more than 10 minutes.
    for _ in range(160):
        response = aws(
            "ecs",
            "describe-tasks",
            "--cluster",
            values["fargate_training_cluster_arn"],
            "--tasks",
            task_arn,
        )
        if response.get("failures"):
            raise RuntimeError(f"ECS describe failed: {response['failures']}")
        task = response["tasks"][0]
        if task["lastStatus"] == "STOPPED":
            containers = task.get("containers", [])
            if not containers or any(container.get("exitCode") != 0 for container in containers):
                raise RuntimeError(f"Lifecycle failed: {task.get('stoppedReason')}; {containers}")
            print("Lifecycle task completed successfully.")
            return
        time.sleep(15)
    aws(
        "ecs",
        "stop-task",
        "--cluster",
        values["fargate_training_cluster_arn"],
        "--task",
        task_arn,
        "--reason",
        "Lifecycle workflow exceeded its 40 minute task deadline",
    )
    raise TimeoutError("Lifecycle task timed out; inspect CloudWatch and the active release.")


if __name__ == "__main__":
    main()
