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


def print_task_logs(
    log_group: str,
    log_stream: str,
    next_token: str | None,
) -> str | None:
    """Print new CloudWatch messages without changing the task outcome."""
    for _ in range(20):
        arguments = [
            "logs",
            "get-log-events",
            "--log-group-name",
            log_group,
            "--log-stream-name",
            log_stream,
            "--start-from-head",
        ]
        if next_token is not None:
            arguments += ["--next-token", next_token]
        try:
            response = aws(*arguments)
        except subprocess.CalledProcessError as error:
            if "ResourceNotFoundException" not in (error.stderr or ""):
                print(f"CloudWatch logs temporarily unavailable: {error.stderr}", flush=True)
            return next_token
        for event in response.get("events", []):
            print(event["message"], flush=True)
        new_token = response.get("nextForwardToken")
        if new_token is None or new_token == next_token:
            return next_token
        next_token = new_token
    return next_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["train", "auto-retrain", "rollback-model", "refresh", "simulate"]
    )
    parser.add_argument("--estimators", type=int)
    parser.add_argument("--maximum-days", type=int, default=95)
    parser.add_argument("--simulation-mode", choices=["both", "managed", "static"], default="both")
    args = parser.parse_args()
    if not 1 <= args.maximum_days <= 95:
        parser.error("maximum-days must be between 1 and 95")
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
    if args.mode == "simulate":
        command += [
            "--maximum-days",
            str(args.maximum_days),
            "--simulation-mode",
            args.simulation_mode,
        ]
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
    log_group = values["fargate_training_log_group"]
    task_id = task_arn.rsplit("/", 1)[-1]
    log_stream = f"training/training/{task_id}"
    log_token = None
    # Training plus model validation and serving refresh may take more than 10 minutes.
    deadline_minutes = 50 if args.mode == "simulate" else 40
    deadline = time.monotonic() + deadline_minutes * 60
    while time.monotonic() < deadline:
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
        log_token = print_task_logs(log_group, log_stream, log_token)
        if task["lastStatus"] == "STOPPED":
            for _ in range(3):
                time.sleep(3)
                log_token = print_task_logs(log_group, log_stream, log_token)
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
        f"Lifecycle workflow exceeded its {deadline_minutes} minute task deadline",
    )
    raise TimeoutError("Lifecycle task timed out; inspect CloudWatch and the active release.")


if __name__ == "__main__":
    main()
