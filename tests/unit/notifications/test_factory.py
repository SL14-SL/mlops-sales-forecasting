from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.notifications import factory
from mlops_sales_forecasting.notifications.contracts import (
    LifecycleEvent,
    LifecycleEventType,
)
from mlops_sales_forecasting.notifications.dispatcher import (
    NullNotificationSink,
)
from mlops_sales_forecasting.notifications.factory import (
    build_notification_sink,
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
    )


def test_missing_notification_config_uses_null_sink() -> None:
    sink = build_notification_sink(
        {}
    )

    assert isinstance(
        sink,
        NullNotificationSink,
    )


def test_disabled_notifications_use_null_sink() -> None:
    sink = build_notification_sink(
        {
            "notifications": {
                "enabled": False,
            }
        }
    )

    assert isinstance(
        sink,
        NullNotificationSink,
    )


def test_logging_notifications_are_enabled() -> None:
    logger = MagicMock()
    sink = build_notification_sink(
        {
            "notifications": {
                "enabled": True,
                "log_events": True,
                "webhook": {
                    "enabled": False,
                },
            }
        },
        logger=logger,
    )

    sink.notify(
        build_event()
    )

    logger.info.assert_called_once()


def test_enabled_webhook_is_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook_sink = MagicMock()
    webhook_factory = MagicMock(
        return_value=webhook_sink
    )
    monkeypatch.setattr(
        factory,
        "WebhookNotificationSink",
        webhook_factory,
    )

    sink = build_notification_sink(
        {
            "notifications": {
                "enabled": True,
                "log_events": False,
                "webhook": {
                    "enabled": True,
                    "url": (
                        "https://example.com/hook"
                    ),
                    "timeout_seconds": 2.5,
                },
            }
        },
        logger=MagicMock(),
    )
    event = build_event()

    sink.notify(event)

    webhook_factory.assert_called_once_with(
        "https://example.com/hook",
        timeout_seconds=2.5,
    )
    webhook_sink.notify.assert_called_once_with(
        event
    )


def test_enabled_webhook_requires_url() -> None:
    with pytest.raises(
        ValueError,
        match="resolved non-empty URL",
    ):
        build_notification_sink(
            {
                "notifications": {
                    "enabled": True,
                    "webhook": {
                        "enabled": True,
                        "url": "",
                    },
                }
            }
        )


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        True,
        "5",
    ],
)
def test_webhook_requires_valid_timeout(
    timeout: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        build_notification_sink(
            {
                "notifications": {
                    "enabled": True,
                    "log_events": False,
                    "webhook": {
                        "enabled": True,
                        "url": (
                            "https://example.com/hook"
                        ),
                        "timeout_seconds": timeout,
                    },
                }
            }
        )