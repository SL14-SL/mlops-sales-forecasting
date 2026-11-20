"""Wait for ECS rollout metadata to catch up after the services-stable waiter."""

import time
from typing import Any


def wait_for_rollout(
    *,
    ecs: Any,
    cluster: str,
    service: str,
    deployment_id: str,
    previous_task_arns: tuple[str, ...] | None = None,
    max_attempts: int = 50,
    delay_seconds: float = 3,
) -> None:
    """Require the requested completed rollout, tolerating temporarily stale reads."""
    if max_attempts < 1 or delay_seconds < 0:
        raise ValueError("Invalid ECS rollout polling limits.")
    last_state: dict[str, Any] = {}
    for attempt in range(max_attempts):
        response = ecs.describe_services(cluster=cluster, services=[service])
        services = response.get("services", [])
        if response.get("failures") or not services:
            raise RuntimeError("ECS service could not be read while verifying rollout.")
        current = services[0]
        deployments = current.get("deployments", [])
        requested = next((item for item in deployments if item["id"] == deployment_id), None)
        primary = next((item for item in deployments if item["status"] == "PRIMARY"), {})
        last_state = {
            "expected_deployment": deployment_id,
            "primary_deployment": primary.get("id"),
            "rollout_state": primary.get("rolloutState"),
            "reason": primary.get("rolloutStateReason"),
            "requested_state": requested.get("rolloutState") if requested else None,
            "requested_reason": requested.get("rolloutStateReason") if requested else None,
        }
        if requested is not None and requested.get("rolloutState") == "FAILED":
            raise RuntimeError(f"ECS requested deployment failed: {last_state}")
        completed = (
            primary.get("id") == deployment_id and primary.get("rolloutState") == "COMPLETED"
        )
        if completed and previous_task_arns is not None:
            after = ecs.list_tasks(
                cluster=cluster,
                serviceName=service,
                desiredStatus="RUNNING",
            )["taskArns"]
            last_state["running_tasks"] = after
            completed = bool(after) and not set(previous_task_arns).intersection(after)
        if completed:
            return
        if attempt + 1 < max_attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(
        "ECS rollout verification timed out (possible circuit rollback or stale tasks): "
        f"{last_state}"
    )
