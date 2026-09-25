from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ...tracking.aliases import ModelAlias
from ...tracking.promotion_service import (
    PromotionOutcome,
)
from ...tracking.registry import (
    ModelRegistrationResult,
)
from .contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from .policy import validate_task_manifest

Clock = Callable[[], datetime]
ReleaseIdFactory = Callable[[], str]


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _new_release_id() -> str:
    """Return a unique serving-release identifier."""

    return f"release-{uuid4()}"


def _validate_registration(
    registration: ModelRegistrationResult,
) -> tuple[str, str]:
    """Return the registered version and immutable model URI."""

    if not isinstance(
        registration,
        ModelRegistrationResult,
    ):
        raise TypeError(
            "Serving release creation requires "
            "ModelRegistrationResult."
        )

    if (
        not registration.registered
        or registration.model_version is None
        or registration.model_uri is None
    ):
        raise ValueError(
            "Serving release creation requires a "
            "registered model candidate."
        )

    return (
        registration.model_version,
        registration.model_uri,
    )


def _validate_promotion(
    *,
    registration: ModelRegistrationResult,
    promotion: PromotionOutcome,
) -> None:
    """Validate that the registered candidate became champion."""

    if not isinstance(
        promotion,
        PromotionOutcome,
    ):
        raise TypeError(
            "Serving release creation requires "
            "PromotionOutcome."
        )

    if not promotion.decision.promote:
        raise ValueError(
            "Serving release creation requires a "
            "successful champion promotion."
        )

    assignment = promotion.champion_assignment

    if assignment is None:
        raise ValueError(
            "Successful promotion has no champion "
            "alias assignment."
        )

    if assignment.alias is not ModelAlias.CHAMPION:
        raise ValueError(
            "Serving release requires a champion "
            "alias assignment."
        )

    if (
        assignment.model_name
        != registration.model_name
        or assignment.model_version
        != registration.model_version
    ):
        raise ValueError(
            "Champion alias assignment does not match "
            "the registered model candidate."
        )


def _created_at_utc(clock: Clock) -> str:
    """Return a validated timezone-aware creation timestamp."""

    created_at = clock()

    if not isinstance(created_at, datetime):
        raise TypeError(
            "Serving release clock must return datetime."
        )

    if created_at.tzinfo is None:
        raise ValueError(
            "Serving release creation timestamp "
            "must include a timezone."
        )

    return created_at.astimezone(
        UTC
    ).isoformat()


def build_serving_release_manifest(
    *,
    registration: ModelRegistrationResult,
    promotion: PromotionOutcome,
    task_type: TaskType,
    model_type: str,
    artifacts: Mapping[
        str,
        ArtifactReference,
    ],
    metadata: Mapping[str, Any] | None = None,
    dataset_version: str | None = None,
    config_hash: str | None = None,
    git_commit: str | None = None,
    release_id: str | None = None,
    clock: Clock = _utc_now,
    release_id_factory: ReleaseIdFactory = (
        _new_release_id
    ),
) -> ServingReleaseManifest:
    """Build a validated immutable serving manifest."""

    model_version, model_uri = (
        _validate_registration(
            registration
        )
    )
    _validate_promotion(
        registration=registration,
        promotion=promotion,
    )

    if not isinstance(task_type, TaskType):
        raise TypeError(
            "Serving release requires TaskType."
        )

    if (
        not isinstance(model_type, str)
        or not model_type.strip()
    ):
        raise ValueError(
            "Serving release model type must not be empty."
        )

    resolved_release_id = (
        release_id
        if release_id is not None
        else release_id_factory()
    )

    manifest = ServingReleaseManifest(
        schema_version=1,
        release_id=resolved_release_id,
        created_at_utc=_created_at_utc(
            clock
        ),
        task_type=task_type,
        model=ModelReference(
            name=registration.model_name,
            version=model_version,
            run_id=registration.run_id,
            uri=model_uri,
            model_type=model_type,
        ),
        artifacts=dict(artifacts),
        dataset_version=dataset_version,
        config_hash=config_hash,
        git_commit=git_commit,
        metadata=(
            dict(metadata)
            if metadata is not None
            else None
        ),
    )

    validate_task_manifest(manifest)

    return manifest