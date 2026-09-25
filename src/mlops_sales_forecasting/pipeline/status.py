from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum


class PipelineRunStatus(StrEnum):
    """Possible states of a training-pipeline run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _validate_timestamp(
    *,
    name: str,
    value: datetime,
) -> None:
    if not isinstance(value, datetime):
        raise TypeError(
            f"{name} must be a datetime."
        )

    if value.tzinfo is None:
        raise ValueError(
            f"{name} must include timezone information."
        )


@dataclass(frozen=True)
class PipelineRun:
    """Immutable lifecycle state of one training run."""

    run_id: str
    status: PipelineRunStatus
    started_at_utc: datetime
    completed_at_utc: datetime | None = None
    rejection_reasons: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(
                "Pipeline run ID must be a non-empty string."
            )

        if not isinstance(
            self.status,
            PipelineRunStatus,
        ):
            raise TypeError(
                "Pipeline run status is invalid."
            )

        _validate_timestamp(
            name="started_at_utc",
            value=self.started_at_utc,
        )

        if self.completed_at_utc is not None:
            _validate_timestamp(
                name="completed_at_utc",
                value=self.completed_at_utc,
            )

            if (
                self.completed_at_utc
                < self.started_at_utc
            ):
                raise ValueError(
                    "Pipeline completion timestamp "
                    "cannot precede its start timestamp."
                )

        if not all(
            isinstance(reason, str) and reason
            for reason in self.rejection_reasons
        ):
            raise ValueError(
                "Pipeline rejection reasons must be "
                "non-empty strings."
            )

        self._validate_state()

    def _validate_state(self) -> None:
        if self.status is PipelineRunStatus.RUNNING:
            if self.completed_at_utc is not None:
                raise ValueError(
                    "Running pipeline cannot have "
                    "a completion timestamp."
                )

            if self.rejection_reasons:
                raise ValueError(
                    "Running pipeline cannot have "
                    "rejection reasons."
                )

            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "Running pipeline cannot have "
                    "error information."
                )

            return

        if self.completed_at_utc is None:
            raise ValueError(
                "Finished pipeline must have "
                "a completion timestamp."
            )

        if self.status is PipelineRunStatus.SUCCEEDED:
            if self.rejection_reasons:
                raise ValueError(
                    "Successful pipeline cannot have "
                    "rejection reasons."
                )

            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "Successful pipeline cannot have "
                    "error information."
                )

        elif self.status is PipelineRunStatus.REJECTED:
            if not self.rejection_reasons:
                raise ValueError(
                    "Rejected pipeline must contain "
                    "at least one reason."
                )

            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "Rejected pipeline cannot have "
                    "error information."
                )

        elif self.status is PipelineRunStatus.FAILED:
            if self.rejection_reasons:
                raise ValueError(
                    "Failed pipeline cannot have "
                    "rejection reasons."
                )

            if not self.error_type or not self.error_message:
                raise ValueError(
                    "Failed pipeline must contain "
                    "error information."
                )

    @classmethod
    def start(
        cls,
        run_id: str,
        *,
        started_at_utc: datetime | None = None,
    ) -> "PipelineRun":
        """Create a running pipeline state."""
        return cls(
            run_id=run_id,
            status=PipelineRunStatus.RUNNING,
            started_at_utc=(
                started_at_utc or _utc_now()
            ),
        )

    def succeed(
        self,
        *,
        completed_at_utc: datetime | None = None,
    ) -> "PipelineRun":
        """Complete a running pipeline successfully."""
        self._require_running()

        return replace(
            self,
            status=PipelineRunStatus.SUCCEEDED,
            completed_at_utc=(
                completed_at_utc or _utc_now()
            ),
        )

    def reject(
        self,
        reasons: tuple[str, ...],
        *,
        completed_at_utc: datetime | None = None,
    ) -> "PipelineRun":
        """Complete a run with a rejected model candidate."""
        self._require_running()

        return replace(
            self,
            status=PipelineRunStatus.REJECTED,
            completed_at_utc=(
                completed_at_utc or _utc_now()
            ),
            rejection_reasons=reasons,
        )

    def fail(
        self,
        error: Exception,
        *,
        completed_at_utc: datetime | None = None,
    ) -> "PipelineRun":
        """Complete a run with technical failure information."""
        self._require_running()

        return replace(
            self,
            status=PipelineRunStatus.FAILED,
            completed_at_utc=(
                completed_at_utc or _utc_now()
            ),
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def _require_running(self) -> None:
        if self.status is not PipelineRunStatus.RUNNING:
            raise ValueError(
                "Only a running pipeline can be completed."
            )

    @property
    def duration_seconds(self) -> float | None:
        """Return the completed run duration."""
        if self.completed_at_utc is None:
            return None

        return (
            self.completed_at_utc
            - self.started_at_utc
        ).total_seconds()