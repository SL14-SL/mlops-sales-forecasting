import json
import logging
import math
from dataclasses import asdict, dataclass
from typing import Literal

PredictionStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class PredictionEvent:
    """Privacy-safe metadata describing one prediction request."""

    request_id: str
    release_id: str
    task_type: str
    model_name: str
    model_version: str
    batch_size: int
    duration_ms: float
    status: PredictionStatus
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")

        if not self.release_id:
            raise ValueError("release_id must not be empty")

        if not self.task_type:
            raise ValueError("task_type must not be empty")

        if not self.model_name:
            raise ValueError("model_name must not be empty")

        if not self.model_version:
            raise ValueError("model_version must not be empty")

        if self.batch_size < 0:
            raise ValueError("batch_size must not be negative")

        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError(
                "duration_ms must be a finite, non-negative number"
            )

        if self.status == "success" and self.error_type is not None:
            raise ValueError(
                "Successful prediction events cannot have an error_type"
            )

        if self.status == "error" and not self.error_type:
            raise ValueError(
                "Failed prediction events must have an error_type"
            )

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Return the event as a serializable dictionary."""
        return asdict(self)


def log_prediction_event(
    event: PredictionEvent,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Write one structured prediction event to the application log."""
    target_logger = logger or logging.getLogger(__name__)
    payload = event.to_dict()

    target_logger.info(
        "prediction_event=%s",
        json.dumps(payload, sort_keys=True),
        extra={"prediction_event": payload},
    )