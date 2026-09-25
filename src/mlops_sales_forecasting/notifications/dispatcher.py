import logging
from collections.abc import Iterable

from .contracts import (
    LifecycleEvent,
    NotificationSink,
)


class NullNotificationSink:
    """Discard lifecycle notifications."""

    def notify(
        self,
        event: LifecycleEvent,
    ) -> None:
        del event


class LoggingNotificationSink:
    """Write lifecycle notifications to application logs."""

    def __init__(
        self,
        logger: logging.Logger,
    ) -> None:
        self._logger = logger

    def notify(
        self,
        event: LifecycleEvent,
    ) -> None:
        self._logger.info(
            "Model lifecycle event | "
            "event_type=%s | "
            "project=%s | "
            "environment=%s | "
            "run_id=%s | "
            "message=%s | "
            "details=%s",
            event.event_type.value,
            event.project_slug,
            event.environment,
            event.run_id,
            event.message,
            dict(event.details),
        )


class NotificationDispatcher:
    """Deliver lifecycle events to configured notification sinks."""

    def __init__(
        self,
        sinks: Iterable[NotificationSink],
        *,
        logger: logging.Logger,
        fail_on_error: bool = False,
    ) -> None:
        self._sinks = tuple(sinks)
        self._logger = logger
        self._fail_on_error = fail_on_error

    def notify(
        self,
        event: LifecycleEvent,
    ) -> None:
        for sink in self._sinks:
            try:
                sink.notify(event)
            except Exception:
                self._logger.exception(
                    "Lifecycle notification delivery failed | "
                    "event_type=%s | sink=%s",
                    event.event_type.value,
                    type(sink).__name__,
                )

                if self._fail_on_error:
                    raise