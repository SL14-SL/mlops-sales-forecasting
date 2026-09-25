from datetime import UTC, datetime

import pytest

from mlops_sales_forecasting.notifications.contracts import (
    LifecycleEvent,
    LifecycleEventType,
    NotificationSink,
)


def build_event() -> LifecycleEvent:
    return LifecycleEvent(
        event_type=(
            LifecycleEventType.CHAMPION_PROMOTED
        ),
        occurred_at_utc=datetime(
            2026,
            9,
            24,
            8,
            30,
            tzinfo=UTC,
        ),
        project_slug="example-project",
        environment="test",
        run_id="run-123",
        message="Candidate promoted to Champion.",
        details={
            "model_version": "7",
            "metric_name": "roc_auc",
        },
    )


def test_lifecycle_event_contains_context() -> None:
    event = build_event()

    assert event.event_type == (
        LifecycleEventType.CHAMPION_PROMOTED
    )
    assert event.project_slug == "example-project"
    assert event.environment == "test"
    assert event.run_id == "run-123"
    assert event.details["model_version"] == "7"


@pytest.mark.parametrize(
    "field_name",
    [
        "project_slug",
        "environment",
        "run_id",
        "message",
    ],
)
def test_lifecycle_event_rejects_empty_context(
    field_name: str,
) -> None:
    values = {
        "event_type": (
            LifecycleEventType.CANDIDATE_REJECTED
        ),
        "occurred_at_utc": datetime.now(UTC),
        "project_slug": "example-project",
        "environment": "test",
        "run_id": "run-123",
        "message": "Candidate rejected.",
    }
    values[field_name] = " "

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        LifecycleEvent(**values)


def test_lifecycle_event_requires_timezone() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        LifecycleEvent(
            event_type=(
                LifecycleEventType.PIPELINE_FAILED
            ),
            occurred_at_utc=datetime(
                2026,
                9,
                24,
                8,
                30,
            ),
            project_slug="example-project",
            environment="test",
            run_id="run-123",
            message="Training failed.",
        )


def test_notification_sink_is_runtime_checkable() -> None:
    class RecordingSink:
        def notify(
            self,
            event: LifecycleEvent,
        ) -> None:
            del event

    assert isinstance(
        RecordingSink(),
        NotificationSink,
    )