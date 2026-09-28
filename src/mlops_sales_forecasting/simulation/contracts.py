import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SimulationDayResult:
    """Operational result recorded for one simulation day."""

    day: int
    cumulative_days: int
    scenario: str
    retraining_enabled: bool
    drift_start_day: int
    drift_duration_days: int
    maximum_base_uplift: float
    maximum_promo_uplift: float
    rmse: float | None = None
    mae: float | None = None
    bias: float | None = None
    n_samples: int | None = None
    window_start: str | None = None
    window_end: str | None = None
    event: str | None = None
    champion_promoted: bool = False
    candidate_run_id: str | None = None
    latest_batch_file: str | None = None

    def __post_init__(self) -> None:
        if self.day < 1:
            raise ValueError("Simulation day must be positive.")

        if self.cumulative_days < 1:
            raise ValueError("Cumulative simulation days must be positive.")

        if not self.scenario:
            raise ValueError("Simulation scenario must not be empty.")

        if self.drift_start_day < 1:
            raise ValueError("Drift start day must be positive.")

        if self.drift_duration_days < 1:
            raise ValueError("Drift duration must be positive.")

        for name in (
            "rmse",
            "mae",
            "bias",
        ):
            value = getattr(
                self,
                name,
            )

            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Simulation metric '{name}' must be finite.")

        if self.n_samples is not None and self.n_samples < 0:
            raise ValueError("Simulation sample count must not be negative.")

    def to_record(
        self,
    ) -> dict[str, Any]:
        """Return a CSV-compatible result record."""
        return asdict(self)
