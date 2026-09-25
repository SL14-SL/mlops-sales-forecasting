from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .contracts import (
    LifecycleEvent,
    LifecycleEventType,
)


def _required_string(
    section: Mapping[str, Any],
    name: str,
    *,
    source: str,
) -> str:
    value = section.get(name)

    if (
        not isinstance(value, str)
        or not value.strip()
        or value.startswith("${")
    ):
        raise ValueError(
            f"{source} '{name}' must be a "
            "resolved non-empty string."
        )

    return value.strip()


def build_lifecycle_event(
    *,
    config: Mapping[str, Any],
    event_type: LifecycleEventType,
    run_id: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    occurred_at_utc: datetime | None = None,
) -> LifecycleEvent:
    """Build a lifecycle event from resolved project configuration."""

    project = config.get("project")

    if not isinstance(project, Mapping):
        raise ValueError(
            "Config must contain a valid "
            "'project' section."
        )

    project_slug = _required_string(
        project,
        "slug",
        source="Project config",
    )
    environment = _required_string(
        config,
        "environment",
        source="Config",
    )

    return LifecycleEvent(
        event_type=event_type,
        occurred_at_utc=(
            occurred_at_utc
            or datetime.now(UTC)
        ),
        project_slug=project_slug,
        environment=environment,
        run_id=run_id,
        message=message,
        details=details or {},
    )