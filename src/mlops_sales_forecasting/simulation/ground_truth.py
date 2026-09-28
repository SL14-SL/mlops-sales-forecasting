from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DriftScenario:
    """Controlled change applied to simulated daily sales."""

    name: str = "stable"
    drift_start_day: int = 46
    drift_duration_days: int = 14
    maximum_base_uplift: float = 0.10
    maximum_promo_uplift: float = 0.35

    def __post_init__(self) -> None:
        if self.name not in {
            "stable",
            "gradual_promo_shift",
        }:
            raise ValueError(f"Unsupported drift scenario: {self.name}")

        if self.drift_start_day < 1:
            raise ValueError("Drift start day must be positive.")

        if self.drift_duration_days < 1:
            raise ValueError("Drift duration must be positive.")

        if self.maximum_base_uplift <= -1:
            raise ValueError("Maximum base uplift must be greater than -1.")

        if self.maximum_promo_uplift <= -1:
            raise ValueError("Maximum promo uplift must be greater than -1.")


@dataclass(frozen=True)
class DriftApplication:
    """Metadata describing one applied simulation regime."""

    progress: float
    base_multiplier: float
    promo_multiplier: float


def calculate_drift_application(
    *,
    current_day: int,
    scenario: DriftScenario,
) -> DriftApplication:
    """Calculate drift strength and sales multipliers."""
    if current_day < 1:
        raise ValueError("Simulation day must be positive.")

    if scenario.name == "stable":
        return DriftApplication(
            progress=0.0,
            base_multiplier=1.0,
            promo_multiplier=1.0,
        )

    if current_day < scenario.drift_start_day:
        progress = 0.0
    else:
        elapsed_days = current_day - scenario.drift_start_day + 1
        progress = min(
            elapsed_days / scenario.drift_duration_days,
            1.0,
        )

    return DriftApplication(
        progress=float(progress),
        base_multiplier=float(1.0 + scenario.maximum_base_uplift * progress),
        promo_multiplier=float(1.0 + scenario.maximum_promo_uplift * progress),
    )


def apply_drift_scenario(
    batch: pd.DataFrame,
    *,
    current_day: int,
    scenario: DriftScenario,
) -> tuple[
    pd.DataFrame,
    DriftApplication,
]:
    """Apply a controlled drift scenario to one daily batch."""
    required_columns = {
        "Sales",
        "Promo",
    }
    missing_columns = required_columns - set(batch.columns)

    if missing_columns:
        raise KeyError(f"Simulation batch is missing columns: {sorted(missing_columns)}.")

    application = calculate_drift_application(
        current_day=current_day,
        scenario=scenario,
    )
    result = batch.copy()

    sales = pd.to_numeric(
        result["Sales"],
        errors="raise",
    ).astype(float)
    promo = pd.to_numeric(
        result["Promo"],
        errors="coerce",
    ).fillna(0)

    multipliers = pd.Series(
        application.base_multiplier,
        index=result.index,
        dtype=float,
    )
    multipliers.loc[promo.eq(1)] = application.promo_multiplier

    result["Sales"] = sales.mul(multipliers).round().astype(int)

    return result, application
