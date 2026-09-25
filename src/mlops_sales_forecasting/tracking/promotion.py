import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class MetricDirection(StrEnum):
    """Direction in which a model metric improves."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class PromotionPolicy:
    """Metric policy used to compare candidate and champion."""

    metric_name: str
    direction: MetricDirection
    minimum_improvement: float = 0.0
    allow_initial_champion: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.metric_name, str)
            or not self.metric_name
        ):
            raise ValueError(
                "Promotion metric name must not be empty."
            )

        if not isinstance(
            self.direction,
            MetricDirection,
        ):
            raise TypeError(
                "Promotion metric direction is invalid."
            )

        if (
            isinstance(
                self.minimum_improvement,
                bool,
            )
            or not isinstance(
                self.minimum_improvement,
                (int, float),
            )
            or not math.isfinite(
                float(self.minimum_improvement)
            )
            or self.minimum_improvement < 0
        ):
            raise ValueError(
                "Minimum improvement must be "
                "a finite non-negative number."
            )

        if not isinstance(
            self.allow_initial_champion,
            bool,
        ):
            raise TypeError(
                "allow_initial_champion must be a boolean."
            )


@dataclass(frozen=True)
class PromotionDecision:
    """Result of comparing a candidate with the champion."""

    promote: bool
    metric_name: str
    candidate_value: float
    champion_value: float | None
    improvement: float | None
    reason: str


def _metric_value(
    metrics: Mapping[str, float],
    metric_name: str,
    *,
    source: str,
) -> float:
    """Return one validated model metric."""
    if metric_name not in metrics:
        raise ValueError(
            f"{source} metrics do not contain "
            f"promotion metric '{metric_name}'."
        )

    value = metrics[metric_name]

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{source} promotion metric "
            f"'{metric_name}' must be finite."
        )

    return float(value)


def evaluate_promotion(
    *,
    candidate_metrics: Mapping[str, float],
    champion_metrics: Mapping[str, float] | None,
    policy: PromotionPolicy,
) -> PromotionDecision:
    """Evaluate whether a candidate should become champion."""
    if not isinstance(policy, PromotionPolicy):
        raise TypeError(
            "Promotion evaluation requires PromotionPolicy."
        )

    candidate_value = _metric_value(
        candidate_metrics,
        policy.metric_name,
        source="Candidate",
    )

    if champion_metrics is None:
        return PromotionDecision(
            promote=policy.allow_initial_champion,
            metric_name=policy.metric_name,
            candidate_value=candidate_value,
            champion_value=None,
            improvement=None,
            reason=(
                "No champion exists; candidate may "
                "become the initial champion."
                if policy.allow_initial_champion
                else (
                    "No champion exists and automatic "
                    "initial promotion is disabled."
                )
            ),
        )

    champion_value = _metric_value(
        champion_metrics,
        policy.metric_name,
        source="Champion",
    )

    if (
        policy.direction
        is MetricDirection.MAXIMIZE
    ):
        improvement = (
            candidate_value - champion_value
        )
    else:
        improvement = (
            champion_value - candidate_value
        )

    promote = improvement > policy.minimum_improvement or math.isclose(
        improvement,
        policy.minimum_improvement,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )   

    return PromotionDecision(
        promote=promote,
        metric_name=policy.metric_name,
        candidate_value=candidate_value,
        champion_value=champion_value,
        improvement=improvement,
        reason=(
            "Candidate satisfies the configured "
            "promotion threshold."
            if promote
            else (
                "Candidate does not satisfy the "
                "configured promotion threshold."
            )
        ),
    )