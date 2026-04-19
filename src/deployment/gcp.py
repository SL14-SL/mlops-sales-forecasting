from __future__ import annotations

import os
import shlex
import subprocess
from typing import Mapping

from src.deployment.config import (
    DeploymentConfigError,
    build_image_uri,
    get_gcp_project_id,
    get_gcp_region,
    get_service_dockerfile,
    get_service_name,
    get_service_runtime_config,
)


class GCPDeploymentError(Exception):
    """Raised when a GCP deployment step fails."""


def _run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run a shell command safely and raise a readable error on failure.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            env=merged_env,
            cwd=cwd,
            capture_output=capture_output,
        )
        return result
    except subprocess.CalledProcessError as exc:
        rendered_cmd = shlex.join(command)
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        details = "\n".join(part for part in [stdout, stderr] if part)
        message = f"Command failed: {rendered_cmd}"
        if details:
            message = f"{message}\n{details}"
        raise GCPDeploymentError(message) from exc
    except FileNotFoundError as exc:
        raise GCPDeploymentError(
            f"Required executable not found while running: {shlex.join(command)}"
        ) from exc


def configure_docker_auth(region: str | None = None) -> None:
    """
    Configure Docker auth for Artifact Registry in the target region.
    """
    target_region = region or get_gcp_region()
    registry_host = f"{target_region}-docker.pkg.dev"

    _run_command(
        ["gcloud", "auth", "configure-docker", registry_host, "--quiet"]
    )


def build_and_push_image(
    service_key: str,
    image_tag: str,
    *,
    project_dir: str = ".",
) -> str:
    """
    Build and push the container image for a configured service.
    Returns the full image URI.
    """
    if not image_tag:
        raise GCPDeploymentError("image_tag must not be empty")

    dockerfile = get_service_dockerfile(service_key)
    image_uri = build_image_uri(service_key, image_tag)

    configure_docker_auth()

    _run_command(
        [
            "docker",
            "build",
            "-f",
            dockerfile,
            "-t",
            image_uri,
            project_dir,
        ]
    )

    _run_command(["docker", "push", image_uri])

    return image_uri


def _flatten_env_vars(env_vars: Mapping[str, str] | None) -> str | None:
    if not env_vars:
        return None

    cleaned: list[str] = []
    for key, value in env_vars.items():
        if value is None:
            continue
        cleaned.append(f"{key}={value}")

    if not cleaned:
        return None

    return ",".join(cleaned)


def deploy_cloud_run_service(
    service_key: str,
    image_tag: str,
    *,
    env_vars: Mapping[str, str] | None = None,
    project_id: str | None = None,
    region: str | None = None,
    service_account: str | None = None,
) -> str:
    """
    Deploy a configured service to Cloud Run.
    Returns the deployed image URI.
    """
    resolved_project_id = project_id or get_gcp_project_id()
    resolved_region = region or get_gcp_region()
    service_name = get_service_name(service_key)
    runtime_cfg = get_service_runtime_config(service_key)
    image_uri = build_image_uri(service_key, image_tag)

    command = [
        "gcloud",
        "run",
        "deploy",
        service_name,
        "--image",
        image_uri,
        "--project",
        resolved_project_id,
        "--region",
        resolved_region,
        "--platform",
        "managed",
        "--quiet",
    ]

    cpu = runtime_cfg.get("cpu")
    if cpu:
        command.extend(["--cpu", str(cpu)])

    memory = runtime_cfg.get("memory")
    if memory:
        command.extend(["--memory", str(memory)])

    allow_unauthenticated = runtime_cfg.get("allow_unauthenticated", True)
    if allow_unauthenticated:
        command.append("--allow-unauthenticated")
    else:
        command.append("--no-allow-unauthenticated")

    env_vars_arg = _flatten_env_vars(env_vars)
    if env_vars_arg:
        command.extend(["--set-env-vars", env_vars_arg])

    if service_account: 
        command.extend(["--service-account", service_account])
    
    port = runtime_cfg.get("port")
    if port:
        command.extend(["--port", str(port)])
    _run_command(command)

    return image_uri


def build_push_and_deploy_cloud_run(
    service_key: str,
    image_tag: str,
    *,
    env_vars: Mapping[str, str] | None = None,
    project_dir: str = ".",
    project_id: str | None = None,
    region: str | None = None,
) -> str:
    """
    Convenience helper for local/manual deploy usage.
    """
    image_uri = build_and_push_image(
        service_key=service_key,
        image_tag=image_tag,
        project_dir=project_dir,
    )

    deploy_cloud_run_service(
        service_key=service_key,
        image_tag=image_tag,
        env_vars=env_vars,
        project_id=project_id,
        region=region,
    )

    return image_uri


def validate_service_exists(service_key: str) -> None:
    """
    Small helper to fail early with a clean config error.
    """
    try:
        get_service_name(service_key)
    except DeploymentConfigError as exc:
        raise GCPDeploymentError(str(exc)) from exc