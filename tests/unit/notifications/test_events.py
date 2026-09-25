from datetime import UTC, datetime

import pytest

from mlops_sales_forecasting.notifications.contracts import (
    LifecycleEventType,
)
from mlops_sales_forecasting.notifications.events import (
    build_lifecycle_event,
)


def config() -> dict:
    return {
        "project": {
            "slug": "example-project",
        },
        "environment": "staging",
    }


def test_builds_event_from_config() -> None:
    timestamp = datetime(
        2026,
        9,
        24,
        11,
        0,
        tzinfo=UTC,
    )

    event = build_lifecycle_event(
        config=config(),
        event_type=(
            LifecycleEventType.CHAMPION_PROMOTED
        ),
        run_id="run-123",
        message="Candidate promoted.",
        details={
            "model_version": "8",
        },
        occurred_at_utc=timestamp,
    )

    assert event.project_slug == "example-project"
    assert event.environment == "staging"
    assert event.run_id == "run-123"
    assert event.occurred_at_utc == timestamp
    assert event.details == {
        "model_version": "8",
    }


def test_requires_project_section() -> None:
    with pytest.raises(
        ValueError,
        match="project",
    ):
        build_lifecycle_event(
            config={
                "environment": "test",
            },
            event_type=(
                LifecycleEventType.PIPELINE_FAILED
            ),
            run_id="run-123",
            message="Training failed.",
        )


@pytest.mark.parametrize(
    ("config_value", "match"),
    [
        (
            {
                "project": {
                    "slug": "",
                },
                "environment": "test",
            },
            "slug",
        ),
        (
            {
                "project": {
                    "slug": "example-project",
                },
                "environment": "${APP_ENV}",
            },
            "environment",
        ),
    ],
)
def test_requires_resolved_context(
    config_value: dict,
    match: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=match,
    ):
        build_lifecycle_event(
            config=config_value,
            event_type=(
                LifecycleEventType.PIPELINE_FAILED
            ),
            run_id="run-123",
            message="Training failed.",
        )