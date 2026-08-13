from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricComparison:
    champion: float
    candidate: float
    change: float
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    checks: dict[str, MetricComparison]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "checks": {
                name: comparison.to_dict()
                for name, comparison
                in self.checks.items()
            },
            "reasons": list(self.reasons),
        }


def relative_change(
    *,
    candidate: float,
    champion: float,
) -> float:
    if champion == 0:
        raise ValueError(
            "Cannot calculate relative change "
            "against zero Champion metric."
        )

    return (
        float(candidate) - float(champion)
    ) / abs(float(champion))


def evaluate_promotion_policy(
    *,
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    validation_rows: int,
    config: dict[str, Any],
) -> PromotionDecision:
    """
    Evaluate a Candidate against the Champion using explicit promotion gates.
    """
    minimum_rows = int(
        config.get(
            "minimum_validation_rows",
            1000,
        )
    )

    if validation_rows < minimum_rows:
        raise ValueError(
            "Not enough validation rows for promotion: "
            f"{validation_rows}/{minimum_rows}"
        )

    required_metric_names = {
        "overall_rmse",
        "promo_rmse",
        "non_promo_rmse",
        "overall_bias",
    }

    missing_candidate = (
        required_metric_names
        - candidate_metrics.keys()
    )
    missing_champion = (
        required_metric_names
        - champion_metrics.keys()
    )

    if missing_candidate:
        raise ValueError(
            "Candidate promotion metrics missing: "
            f"{sorted(missing_candidate)}"
        )

    if missing_champion:
        raise ValueError(
            "Champion promotion metrics missing: "
            f"{sorted(missing_champion)}"
        )

    checks: dict[str, MetricComparison] = {}
    reasons: list[str] = []

    minimum_improvement = float(
        config.get(
            "minimum_relative_rmse_improvement",
            0.005,
        )
    )

    overall_change = relative_change(
        candidate=candidate_metrics[
            "overall_rmse"
        ],
        champion=champion_metrics[
            "overall_rmse"
        ],
    )

    overall_passed = (
        overall_change
        <= -minimum_improvement
    )

    overall_reason = (
        "Candidate satisfies minimum overall "
        "RMSE improvement."
        if overall_passed
        else (
            "Candidate does not satisfy minimum "
            "overall RMSE improvement."
        )
    )

    checks["overall_rmse"] = MetricComparison(
        champion=float(
            champion_metrics["overall_rmse"]
        ),
        candidate=float(
            candidate_metrics["overall_rmse"]
        ),
        change=overall_change,
        passed=overall_passed,
        reason=overall_reason,
    )

    if not overall_passed:
        reasons.append(overall_reason)

    maximum_segment_regression = float(
        config.get(
            "maximum_segment_rmse_regression",
            0.02,
        )
    )

    required_segments = config.get(
        "required_segments",
        [
            "promo",
            "non_promo",
        ],
    )

    for segment in required_segments:
        metric_name = f"{segment}_rmse"

        segment_change = relative_change(
            candidate=candidate_metrics[
                metric_name
            ],
            champion=champion_metrics[
                metric_name
            ],
        )

        segment_passed = (
            segment_change
            <= maximum_segment_regression
        )

        segment_reason = (
            f"Segment '{segment}' stays within "
            "the allowed RMSE regression."
            if segment_passed
            else (
                f"Segment '{segment}' exceeds "
                "the allowed RMSE regression."
            )
        )

        checks[metric_name] = (
            MetricComparison(
                champion=float(
                    champion_metrics[
                        metric_name
                    ]
                ),
                candidate=float(
                    candidate_metrics[
                        metric_name
                    ]
                ),
                change=(
                    segment_change
                ),
                passed=segment_passed,
                reason=segment_reason,
            )
        )

        if not segment_passed:
            reasons.append(segment_reason)

    maximum_bias_regression = float(
        config.get(
            "maximum_absolute_bias_regression",
            100.0,
        )
    )

    candidate_abs_bias = abs(
        float(
            candidate_metrics[
                "overall_bias"
            ]
        )
    )
    champion_abs_bias = abs(
        float(
            champion_metrics[
                "overall_bias"
            ]
        )
    )

    bias_regression = (
        candidate_abs_bias
        - champion_abs_bias
    )

    bias_passed = (
        bias_regression
        <= maximum_bias_regression
    )

    bias_reason = (
        "Candidate bias stays within the "
        "allowed regression."
        if bias_passed
        else (
            "Candidate absolute bias exceeds "
            "the allowed regression."
        )
    )

    checks["overall_bias"] = MetricComparison(
        champion=champion_abs_bias,
        candidate=candidate_abs_bias,
        change=bias_regression,
        passed=bias_passed,
        reason=bias_reason,
    )

    if not bias_passed:
        reasons.append(bias_reason)

    accepted = all(
        check.passed
        for check in checks.values()
    )

    return PromotionDecision(
        accepted=accepted,
        checks=checks,
        reasons=reasons,
    )