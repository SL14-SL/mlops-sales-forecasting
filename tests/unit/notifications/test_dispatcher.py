from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.notifications.contracts import (
    LifecycleEvent,
    LifecycleEventType,
)
from mlops_sales_forecasting.notifications.dispatcher import (
    LoggingNotificationSink,
    NotificationDispatcher,
    NullNotificationSink,
)


def build_event() -> LifecycleEvent:
    return LifecycleEvent(
        event_type=(
            LifecycleEventType.CHALLENGER_REGISTERED
        ),
        occurred_at_utc=datetime.now(UTC),
        project_slug="example-project",
        environment="test",
        run_id="run-123",
        message="Challenger registered.",
        details={
            "model_version": "4",
        },
    )


def test_null_sink_accepts_event() -> None:
    NullNotificationSink().notify(
        build_event()
    )


def test_logging_sink_writes_event() -> None:
    logger = MagicMock()

    LoggingNotificationSink(
        logger
    ).notify(
        build_event()
    )

    logger.info.assert_called_once()


def test_dispatcher_delivers_to_all_sinks() -> None:
    first_sink = MagicMock()
    second_sink = MagicMock()
    event = build_event()

    dispatcher = NotificationDispatcher(
        [
            first_sink,
            second_sink,
        ],
        logger=MagicMock(),
    )

    dispatcher.notify(event)

    first_sink.notify.assert_called_once_with(
        event
    )
    second_sink.notify.assert_called_once_with(
        event
    )


def test_dispatcher_continues_after_delivery_failure() -> None:
    failing_sink = MagicMock()
    failing_sink.notify.side_effect = RuntimeError(
        "webhook unavailable"
    )
    healthy_sink = MagicMock()
    logger = MagicMock()
    event = build_event()

    dispatcher = NotificationDispatcher(
        [
            failing_sink,
            healthy_sink,
        ],
        logger=logger,
    )

    dispatcher.notify(event)

    healthy_sink.notify.assert_called_once_with(
        event
    )
    logger.exception.assert_called_once()


def test_dispatcher_can_fail_on_delivery_error() -> None:
    failing_sink = MagicMock()
    failing_sink.notify.side_effect = RuntimeError(
        "webhook unavailable"
    )

    dispatcher = NotificationDispatcher(
        [failing_sink],
        logger=MagicMock(),
        fail_on_error=True,
    )

    with pytest.raises(
        RuntimeError,
        match="webhook unavailable",
    ):
        dispatcher.notify(
            build_event()
        )