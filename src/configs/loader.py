from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.configs.paths import get_project_root
from src.configs.environment import _detect_environment, _inject_runtime_env, _override_gcs_bucket_paths, _resolve_env_placeholders


PROJECT_ROOT = get_project_root()

# Load local environment variables, if present.
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config

def get_path(name: str, config_name: str | None = None) -> str:
    """
    Return a configured path from the selected config.

    Raises KeyError if paths.<name> is missing.
    """
    config = load_config(config_name)
    paths = config.get("paths", {})

    if not isinstance(paths, dict):
        raise KeyError("Config does not contain a valid 'paths' section.")

    if name not in paths:
        raise KeyError(f"Path '{name}' not found in config paths.")

    return str(paths[name])


def load_config(config_name: str | None = None) -> dict[str, Any]:
    """
    Load a config file from configs/.

    - If config_name is omitted, use <env>.yaml where env is derived from APP_ENV/K_SERVICE.
    - Resolve ${ENV_VAR} and ${ENV_VAR:-default} placeholders.
    - Apply bucket overrides for gs:// paths when GCS_BUCKET_NAME is set.
    - Inject selected service URLs into os.environ.
    """
    env = _detect_environment()
    file_to_load = config_name if config_name else f"{env}.yaml"
    config_path = PROJECT_ROOT / "configs" / file_to_load

    config = _load_yaml(config_path)
    config = _resolve_env_placeholders(config)
    config = _override_gcs_bucket_paths(config)

    config.setdefault("environment", env)

    _inject_runtime_env(config)
    return config


