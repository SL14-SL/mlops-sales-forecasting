import json
from urllib.parse import urlparse
from urllib.request import (
    Request,
    urlopen,
)

from .contracts import LifecycleEvent


class WebhookNotificationSink:
    """Deliver lifecycle events to an HTTP webhook."""

    def __init__(
        self,
        webhook_url: str,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        parsed_url = urlparse(
            webhook_url
        )

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            raise ValueError(
                "Webhook URL must be a valid HTTP or HTTPS URL."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "Webhook timeout must be greater than zero."
            )

        self._webhook_url = webhook_url
        self._timeout_seconds = timeout_seconds

    def notify(
        self,
        event: LifecycleEvent,
    ) -> None:
        payload = {
            "text": (
                f"[{event.environment}] "
                f"{event.message}"
            ),
            "event_type": event.event_type.value,
            "occurred_at_utc": (
                event.occurred_at_utc.isoformat()
            ),
            "project_slug": event.project_slug,
            "environment": event.environment,
            "run_id": event.run_id,
            "message": event.message,
            "details": dict(event.details),
        }

        request = Request(
            self._webhook_url,
            data=json.dumps(
                payload,
                default=str,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": (
                    "mlops-lifecycle-notifier/1.0"
                ),
            },
            method="POST",
        )

        with urlopen(
            request,
            timeout=self._timeout_seconds,
        ) as response:
            response.read()