import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def get_project_root() -> Path:
    """Find the project root by walking upward until configs/ and src/ exist."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() and (parent / "src").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = get_project_root()

# Load local environment variables, if present.
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_env_placeholders(value: Any) -> Any:
    """
    Recursively resolve ${VAR} and ${VAR:-default} placeholders in YAML content.
    Leaves unresolved placeholders unchanged if no env var/default is available.
    """
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]

    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.getenv(var_name)

            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return match.group(0)

        return _ENV_VAR_PATTERN.sub(replace, value)

    return value


def _detect_environment() -> str:
    """
    Determine active environment.

    Priority:
    1. APP_ENV
    2. K_SERVICE -> prod
    3. dev
    """
    env = os.getenv("APP_ENV")
    if env:
        return env

    if os.getenv("K_SERVICE"):
        return "prod"

    return "dev"


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def _override_gcs_bucket_paths(config: dict[str, Any]) -> dict[str, Any]:
    """
    Override gs:// bucket prefixes in config['paths'] when GCS_BUCKET_NAME is set.

    Example:
      gs://old-bucket/data/raw
    becomes:
      gs://new-bucket/data/raw
    """
    env_bucket = os.getenv("GCS_BUCKET_NAME")
    if not env_bucket:
        return config

    bucket_prefix = "" if env_bucket.startswith("gs://") else "gs://"
    new_base_path = f"{bucket_prefix}{env_bucket}"

    paths = config.get("paths")
    if not isinstance(paths, dict):
        return config

    for key, path in paths.items():
        if isinstance(path, str) and path.startswith("gs://"):
            parts = path.replace("gs://", "", 1).split("/", 1)
            if len(parts) > 1:
                paths[key] = f"{new_base_path}/{parts[1]}"
            else:
                paths[key] = new_base_path

    return config


def _inject_runtime_env(config: dict[str, Any]) -> None:
    """
    Push selected config values into process env for downstream libraries.
    """
    services = config.get("services", {})
    if isinstance(services, dict):
        prefect_api_url = services.get("prefect_api_url")
        if prefect_api_url:
            os.environ.setdefault("PREFECT_API_URL", str(prefect_api_url))

    # Support both nested and legacy placement
    tracking = config.get("tracking", {})
    mlflow_tracking_uri = None

    if isinstance(tracking, dict):
        mlflow_tracking_uri = tracking.get("mlflow_tracking_uri")

    if not mlflow_tracking_uri:
        mlflow_tracking_uri = config.get("mlflow_tracking_uri")

    if mlflow_tracking_uri and "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = str(mlflow_tracking_uri)



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

    # Ensure environment is visible in runtime config when omitted
    config.setdefault("environment", env)

    _inject_runtime_env(config)
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


def file_exists(path: str) -> bool:
    """
    Check existence for local paths and gs:// paths.
    """
    if path.startswith("gs://"):
        try:
            import gcsfs
        except ImportError as exc:
            raise RuntimeError(
                "gcsfs is required for checking gs:// paths."
            ) from exc

        fs = gcsfs.GCSFileSystem()
        return fs.exists(path)

    return Path(path).exists()


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Local paths are created on disk.
    gs:// paths are left untouched because bucket/prefix creation is implicit.
    """
    if path.startswith("gs://"):
        return

    Path(path).mkdir(parents=True, exist_ok=True)