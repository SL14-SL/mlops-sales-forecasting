import logging
from collections.abc import Mapping
from typing import Any

from ..utils.logger import get_logger
from .contracts import NotificationSink
from .dispatcher import (
    LoggingNotificationSink,
    NotificationDispatcher,
    NullNotificationSink,
)
from .webhook import WebhookNotificationSink


def _boolean_setting(
    section: Mapping[str, Any],
    name: str,
    *,
    default: bool,
) -> bool:
    value = section.get(name, default)

    if not isinstance(value, bool):
        raise TypeError(
            f"Notification setting '{name}' must be a boolean."
        )

    return value


def _webhook_timeout(
    section: Mapping[str, Any],
) -> float:
    value = section.get(
        "timeout_seconds",
        5.0,
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ValueError(
            "Notification webhook timeout_seconds "
            "must be greater than zero."
        )

    return float(value)


def build_notification_sink(
    config: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> NotificationSink:
    """Build the configured lifecycle-notification sink."""

    notifications = config.get(
        "notifications"
    )

    if notifications is None:
        return NullNotificationSink()

    if not isinstance(
        notifications,
        Mapping,
    ):
        raise TypeError(
            "Config section 'notifications' "
            "must be a mapping."
        )

    if not _boolean_setting(
        notifications,
        "enabled",
        default=True,
    ):
        return NullNotificationSink()

    target_logger = logger or get_logger(
        __name__
    )
    sinks: list[NotificationSink] = []

    if _boolean_setting(
        notifications,
        "log_events",
        default=True,
    ):
        sinks.append(
            LoggingNotificationSink(
                target_logger
            )
        )

    webhook = notifications.get(
        "webhook",
        {},
    )

    if not isinstance(webhook, Mapping):
        raise TypeError(
            "Notification setting 'webhook' "
            "must be a mapping."
        )

    if _boolean_setting(
        webhook,
        "enabled",
        default=False,
    ):
        webhook_url = webhook.get(
            "url"
        )

        if (
            not isinstance(webhook_url, str)
            or not webhook_url.strip()
            or webhook_url.startswith("${")
        ):
            raise ValueError(
                "Enabled notification webhook requires "
                "a resolved non-empty URL."
            )

        sinks.append(
            WebhookNotificationSink(
                webhook_url.strip(),
                timeout_seconds=(
                    _webhook_timeout(
                        webhook
                    )
                ),
            )
        )

    if not sinks:
        return NullNotificationSink()

    fail_on_error = _boolean_setting(
        notifications,
        "fail_on_error",
        default=False,
    )

    return NotificationDispatcher(
        sinks,
        logger=target_logger,
        fail_on_error=fail_on_error,
    )