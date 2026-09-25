from dataclasses import dataclass
from typing import Any

from .bundle_loader import (
    load_active_serving_bundle,
    load_serving_bundle,
)
from .model_loader import (
    configure_mlflow,
    load_xgboost_model,
)
from .releases.contracts import TaskType
from .serving_bundle import ServingBundle

EXPECTED_TASK_TYPE = TaskType.FORECASTING


@dataclass(frozen=True)
class ServingSettings:
    """Configuration required to load a serving bundle."""

    models_path: str
    tracking_uri: str
    serving_alias: str
    task_type: TaskType = EXPECTED_TASK_TYPE


def _required_config_string(
    section: dict[str, Any],
    name: str,
    *,
    section_name: str,
) -> str:
    """Return a required non-empty configuration string."""
    value = section.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Missing or invalid serving configuration: "
            f"{section_name}.{name}."
        )

    return value


def serving_settings_from_config(
    config: dict[str, Any],
) -> ServingSettings:
    """Build serving settings from the application configuration."""
    paths = config.get("paths")
    tracking = config.get("tracking")
    serving = config.get("serving")

    if not isinstance(paths, dict):
        raise ValueError(
            "Missing or invalid serving configuration: paths."
        )

    if not isinstance(tracking, dict):
        raise ValueError(
            "Missing or invalid serving configuration: tracking."
        )

    if not isinstance(serving, dict):
        raise ValueError(
            "Missing or invalid serving configuration: serving."
        )

    return ServingSettings(
        models_path=_required_config_string(
            paths,
            "models",
            section_name="paths",
        ),
        tracking_uri=_required_config_string(
            tracking,
            "mlflow_tracking_uri",
            section_name="tracking",
        ),
        serving_alias=_required_config_string(
            serving,
            "alias",
            section_name="serving",
        ),
    )


def load_active_bundle(
    config: dict[str, Any],
) -> ServingBundle:
    """Load the active bundle from application configuration."""
    settings = serving_settings_from_config(config)
    configure_mlflow(settings.tracking_uri)

    bundle = load_active_serving_bundle(
        models_path=settings.models_path,
        serving_alias=settings.serving_alias,
        model_loader=load_xgboost_model,
    )

    if bundle.manifest.task_type is not settings.task_type:
        raise ValueError(
            "Loaded serving bundle has an unexpected task type."
        )

    return bundle


def load_bundle_for_release(
    config: dict[str, Any],
    *,
    release_id: str,
) -> ServingBundle:
    """Load a specific serving release from configuration."""
    settings = serving_settings_from_config(config)
    configure_mlflow(settings.tracking_uri)

    bundle = load_serving_bundle(
        models_path=settings.models_path,
        release_id=release_id,
        serving_alias=settings.serving_alias,
        model_loader=load_xgboost_model,
    )

    if bundle.manifest.task_type is not settings.task_type:
        raise ValueError(
            "Loaded serving bundle has an unexpected task type."
        )

    return bundle