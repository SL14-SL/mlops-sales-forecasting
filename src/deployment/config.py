from __future__ import annotations
import os
from pathlib import Path
from typing import Any


import yaml


class DeploymentConfigError(Exception):
    """Raised when deployment configuration is missing or invalid."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)

    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]

    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DeploymentConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise DeploymentConfigError(f"Invalid YAML structure in: {path}")

    return _expand_env_vars(data)


def load_gcp_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load the GCP deployment config from configs/gcp.yaml.
    """
    path = Path(config_path) if config_path else _project_root() / "configs" / "gcp.yaml"
    data = _load_yaml(path)

    if "gcp" not in data:
        raise DeploymentConfigError("Missing top-level 'gcp' key in configs/gcp.yaml")

    gcp_cfg = data["gcp"]
    if not isinstance(gcp_cfg, dict):
        raise DeploymentConfigError("'gcp' must be a dictionary")

    return gcp_cfg


def get_gcp_project_id(config_path: str | None = None) -> str:
    gcp_cfg = load_gcp_config(config_path)
    project_id = gcp_cfg.get("project_id")

    if not project_id:
        raise DeploymentConfigError("Missing 'gcp.project_id' in configs/gcp.yaml")

    return str(project_id)


def get_gcp_region(config_path: str | None = None) -> str:
    gcp_cfg = load_gcp_config(config_path)
    region = gcp_cfg.get("region")

    if not region:
        raise DeploymentConfigError("Missing 'gcp.region' in configs/gcp.yaml")

    return str(region)


def get_artifact_registry_prefix(config_path: str | None = None) -> str:
    gcp_cfg = load_gcp_config(config_path)

    artifact_registry = gcp_cfg.get("artifact_registry", {})
    if not isinstance(artifact_registry, dict):
        raise DeploymentConfigError("'gcp.artifact_registry' must be a dictionary")

    image_prefix = artifact_registry.get("image_prefix")
    if not image_prefix:
        raise DeploymentConfigError(
            "Missing 'gcp.artifact_registry.image_prefix' in configs/gcp.yaml"
        )

    return str(image_prefix).rstrip("/")


def get_cloud_run_services(config_path: str | None = None) -> dict[str, Any]:
    gcp_cfg = load_gcp_config(config_path)

    cloud_run = gcp_cfg.get("cloud_run", {})
    if not isinstance(cloud_run, dict):
        raise DeploymentConfigError("'gcp.cloud_run' must be a dictionary")

    services = cloud_run.get("services")
    if not isinstance(services, dict):
        raise DeploymentConfigError(
            "Missing or invalid 'gcp.cloud_run.services' in configs/gcp.yaml"
        )

    return services


def get_service_config(service_key: str, config_path: str | None = None) -> dict[str, Any]:
    services = get_cloud_run_services(config_path)

    if service_key not in services:
        available = ", ".join(sorted(services.keys()))
        raise DeploymentConfigError(
            f"Unknown service '{service_key}'. Available services: {available}"
        )

    service_cfg = services[service_key]
    if not isinstance(service_cfg, dict):
        raise DeploymentConfigError(
            f"'gcp.cloud_run.services.{service_key}' must be a dictionary"
        )

    return service_cfg


def get_service_name(service_key: str, config_path: str | None = None) -> str:
    service_cfg = get_service_config(service_key, config_path)
    service_name = service_cfg.get("service_name")

    if not service_name:
        raise DeploymentConfigError(
            f"Missing 'service_name' for service '{service_key}' in configs/gcp.yaml"
        )

    return str(service_name)


def get_service_image_name(service_key: str, config_path: str | None = None) -> str:
    service_cfg = get_service_config(service_key, config_path)
    image_name = service_cfg.get("image_name")

    if not image_name:
        raise DeploymentConfigError(
            f"Missing 'image_name' for service '{service_key}' in configs/gcp.yaml"
        )

    return str(image_name)


def get_service_dockerfile(service_key: str, config_path: str | None = None) -> str:
    service_cfg = get_service_config(service_key, config_path)
    dockerfile = service_cfg.get("dockerfile")

    if not dockerfile:
        raise DeploymentConfigError(
            f"Missing 'dockerfile' for service '{service_key}' in configs/gcp.yaml"
        )

    return str(dockerfile)


def build_image_uri(
    service_key: str,
    tag: str,
    config_path: str | None = None,
) -> str:
    if not tag:
        raise DeploymentConfigError("Image tag must not be empty")

    prefix = get_artifact_registry_prefix(config_path)
    image_name = get_service_image_name(service_key, config_path)

    return f"{prefix}/{image_name}:{tag}"


def get_service_runtime_config(
    service_key: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Return runtime-related Cloud Run values for a service.
    Keeps this flexible for later expansion.
    """
    service_cfg = get_service_config(service_key, config_path)

    return {
        "cpu": service_cfg.get("cpu"),
        "memory": service_cfg.get("memory"),
        "port": service_cfg.get("port"),
        "allow_unauthenticated": service_cfg.get("allow_unauthenticated", True),
    }