from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class LifecycleEventType(StrEnum):
    """Model-lifecycle events that may trigger notifications."""

    PIPELINE_FAILED = "pipeline_failed"
    CANDIDATE_REJECTED = "candidate_rejected"
    CHALLENGER_REGISTERED = "challenger_registered"
    CHAMPION_PROMOTED = "champion_promoted"
    SERVING_RELEASE_PUBLISHED = (
        "serving_release_published"
    )


@dataclass(frozen=True)
class LifecycleEvent:
    """Structured notification emitted by the model lifecycle."""

    event_type: LifecycleEventType
    occurred_at_utc: datetime
    project_slug: str
    environment: str
    run_id: str
    message: str
    details: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.occurred_at_utc.tzinfo is None:
            raise ValueError(
                "Lifecycle event timestamp must be timezone-aware."
            )

        for field_name in (
            "project_slug",
            "environment",
            "run_id",
            "message",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Lifecycle event field '{field_name}' "
                    "must be a non-empty string."
                )


@runtime_checkable
class NotificationSink(Protocol):
    """Deliver structured model-lifecycle notifications."""

    def notify(
        self,
        event: LifecycleEvent,
    ) -> None:
        """Deliver one lifecycle event."""
        ...