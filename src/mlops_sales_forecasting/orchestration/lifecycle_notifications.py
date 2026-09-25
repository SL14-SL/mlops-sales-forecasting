from collections.abc import Mapping
from typing import Any

from ..inference.releases.publisher import (
    PublishedServingRelease,
)
from ..notifications.contracts import (
    LifecycleEventType,
    NotificationSink,
)
from ..notifications.events import (
    build_lifecycle_event,
)
from ..tracking.lifecycle import (
    CandidateLifecycleResult,
)


def notify_lifecycle_failure(
    *,
    sink: NotificationSink,
    config: Mapping[str, Any],
    run_id: str,
    error: Exception,
) -> None:
    """Notify recipients about a technical lifecycle failure."""

    sink.notify(
        build_lifecycle_event(
            config=config,
            event_type=(
                LifecycleEventType.PIPELINE_FAILED
            ),
            run_id=run_id,
            message="Model lifecycle failed.",
            details={
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(error),
            },
        )
    )


def notify_candidate_outcome(
    *,
    sink: NotificationSink,
    config: Mapping[str, Any],
    candidate: CandidateLifecycleResult,
    serving_release: (
        PublishedServingRelease | None
    ),
) -> None:
    """Notify recipients about the final candidate outcome."""

    registration = candidate.registration

    if not registration.registered:
        sink.notify(
            build_lifecycle_event(
                config=config,
                event_type=(
                    LifecycleEventType
                    .CANDIDATE_REJECTED
                ),
                run_id=registration.run_id,
                message=(
                    "Model candidate rejected by "
                    "the quality gate."
                ),
                details={
                    "model_name": (
                        registration.model_name
                    ),
                    "reason": registration.reason,
                },
            )
        )
        return

    promotion = candidate.promotion

    if promotion is None:
        raise ValueError(
            "Registered candidate must contain "
            "a promotion outcome."
        )

    decision = promotion.decision
    common_details = {
        "model_name": registration.model_name,
        "model_version": (
            registration.model_version
        ),
        "model_uri": registration.model_uri,
        "metric_name": decision.metric_name,
        "candidate_value": (
            decision.candidate_value
        ),
        "champion_value": (
            decision.champion_value
        ),
        "improvement": decision.improvement,
        "decision_reason": decision.reason,
    }

    if not decision.promote:
        sink.notify(
            build_lifecycle_event(
                config=config,
                event_type=(
                    LifecycleEventType
                    .CHALLENGER_REGISTERED
                ),
                run_id=registration.run_id,
                message=(
                    "Model candidate registered "
                    "as Challenger."
                ),
                details=common_details,
            )
        )
        return

    if serving_release is None:
        raise ValueError(
            "Promoted candidate must contain "
            "a published serving release."
        )

    sink.notify(
        build_lifecycle_event(
            config=config,
            event_type=(
                LifecycleEventType.CHAMPION_PROMOTED
            ),
            run_id=registration.run_id,
            message=(
                "Model candidate promoted to Champion."
            ),
            details={
                **common_details,
                "previous_champion_version": (
                    promotion
                    .previous_champion_version
                ),
            },
        )
    )

    sink.notify(
        build_lifecycle_event(
            config=config,
            event_type=(
                LifecycleEventType
                .SERVING_RELEASE_PUBLISHED
            ),
            run_id=registration.run_id,
            message="Serving release published.",
            details={
                "model_name": (
                    registration.model_name
                ),
                "model_version": (
                    registration.model_version
                ),
                "release_id": (
                    serving_release
                    .manifest
                    .release_id
                ),
                "release_root": (
                    serving_release.release_root
                ),
            },
        )
    )