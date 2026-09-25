from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from mlops_sales_forecasting.configs.environment import (
    detect_environment,
    inject_runtime_env,
    override_gcs_bucket_paths,
    resolve_env_placeholders,
)
from mlops_sales_forecasting.configs.paths import get_project_root

PROJECT_ROOT = get_project_root()


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping: {config_path}"
        )

    return config


def _resolve_config_filename(
    config_name: str | None,
    environment: str,
) -> str:
    """Return a safe YAML filename for the requested configuration."""
    filename = config_name or f"{environment}.yaml"
    candidate = Path(filename)

    if candidate.name != filename:
        raise ValueError(
            "Config name must be a filename without directory components."
        )

    if candidate.suffix not in {".yaml", ".yml"}:
        raise ValueError(
            "Config name must use the .yaml or .yml extension."
        )

    return filename


def load_config(
    config_name: str | None = None,
) -> dict[str, Any]:
    """Load and resolve a project configuration from configs/."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    environment = detect_environment()
    filename = _resolve_config_filename(
        config_name,
        environment,
    )
    config_path = PROJECT_ROOT / "configs" / filename

    config = _load_yaml(config_path)
    resolved_config = resolve_env_placeholders(config)
    resolved_config = override_gcs_bucket_paths(resolved_config)
    resolved_config.setdefault("environment", environment)

    inject_runtime_env(resolved_config)
    return resolved_config


def get_path(
    name: str,
    config_name: str | None = None,
) -> str:
    """Return a named path from the selected configuration."""
    config = load_config(config_name)
    paths = config.get("paths")

    if not isinstance(paths, dict):
        raise KeyError(
            "Config does not contain a valid 'paths' section."
        )

    if name not in paths:
        raise KeyError(
            f"Path '{name}' not found in config paths."
        )

    return str(paths[name])