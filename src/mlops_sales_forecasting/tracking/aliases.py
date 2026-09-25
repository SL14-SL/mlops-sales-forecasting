from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mlflow import MlflowClient

from .registry import ModelRegistrationResult


class ModelAlias(StrEnum):
    """Supported registered-model lifecycle aliases."""

    CHALLENGER = "challenger"
    CHAMPION = "champion"


@dataclass(frozen=True)
class AliasAssignment:
    """Result of assigning a registered-model alias."""

    model_name: str
    model_version: str
    alias: ModelAlias


def _tracking_uri_from_config(
    config: Mapping[str, Any],
) -> str:
    """Return the configured MLflow tracking URI."""
    tracking = config.get("tracking")

    if not isinstance(tracking, Mapping):
        raise ValueError(
            "Config must contain a valid "
            "'tracking' section."
        )

    tracking_uri = tracking.get(
        "mlflow_tracking_uri"
    )

    if (
        not isinstance(tracking_uri, str)
        or not tracking_uri.strip()
        or tracking_uri.startswith("${")
    ):
        raise ValueError(
            "Tracking config 'mlflow_tracking_uri' "
            "must be a resolved non-empty string."
        )

    return tracking_uri


def _require_registered_model(
    registration: ModelRegistrationResult,
) -> tuple[str, str]:
    """Return model identity from a successful registration."""
    if not isinstance(
        registration,
        ModelRegistrationResult,
    ):
        raise TypeError(
            "Alias assignment requires "
            "ModelRegistrationResult."
        )

    if (
        not registration.registered
        or registration.model_version is None
    ):
        raise ValueError(
            "Cannot assign an alias to "
            "an unregistered model candidate."
        )

    return (
        registration.model_name,
        registration.model_version,
    )


def _assign_alias(
    *,
    registration: ModelRegistrationResult,
    config: Mapping[str, Any],
    alias: ModelAlias,
) -> AliasAssignment:
    """Assign one lifecycle alias to a model version."""
    model_name, model_version = (
        _require_registered_model(
            registration
        )
    )
    client = MlflowClient(
        tracking_uri=(
            _tracking_uri_from_config(
                config
            )
        )
    )

    client.set_registered_model_alias(
        name=model_name,
        alias=alias.value,
        version=model_version,
    )

    return AliasAssignment(
        model_name=model_name,
        model_version=model_version,
        alias=alias,
    )


def assign_challenger(
    *,
    registration: ModelRegistrationResult,
    config: Mapping[str, Any],
) -> AliasAssignment:
    """Assign the challenger alias to a candidate."""
    return _assign_alias(
        registration=registration,
        config=config,
        alias=ModelAlias.CHALLENGER,
    )


def promote_to_champion(
    *,
    registration: ModelRegistrationResult,
    config: Mapping[str, Any],
) -> AliasAssignment:
    """Assign champion after an external promotion decision."""
    return _assign_alias(
        registration=registration,
        config=config,
        alias=ModelAlias.CHAMPION,
    )

def restore_champion(
    *,
    registration: ModelRegistrationResult,
    previous_champion_version: str | None,
    config: Mapping[str, Any],
) -> AliasAssignment | None:
    """Restore the champion alias after a failed release."""

    model_name, _ = _require_registered_model(
        registration
    )

    if (
        previous_champion_version is not None
        and (
            not isinstance(
                previous_champion_version,
                str,
            )
            or not previous_champion_version.strip()
        )
    ):
        raise ValueError(
            "Previous champion version must be "
            "a non-empty string or None."
        )

    client = MlflowClient(
        tracking_uri=(
            _tracking_uri_from_config(
                config
            )
        )
    )

    if previous_champion_version is None:
        client.delete_registered_model_alias(
            name=model_name,
            alias=ModelAlias.CHAMPION.value,
        )
        return None

    client.set_registered_model_alias(
        name=model_name,
        alias=ModelAlias.CHAMPION.value,
        version=previous_champion_version,
    )

    return AliasAssignment(
        model_name=model_name,
        model_version=(
            previous_champion_version
        ),
        alias=ModelAlias.CHAMPION,
    )