import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionGuardrailResult:
    """Result of paired business-metric promotion gates."""

    approved: bool
    reasons: tuple[str, ...]
    relative_rmse_improvement: float
    segment_rmse_regressions: Mapping[str, float]
    absolute_bias_regression: float


def _metric(
    metrics: Mapping[str, float],
    name: str,
    *,
    source: str,
) -> float:
    if name not in metrics:
        raise ValueError(f"{source} metrics do not contain '{name}'.")

    value = metrics[name]

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{source} metric '{name}' must be finite.")

    return float(value)


def _non_negative_float(
    config: Mapping[str, Any],
    name: str,
    *,
    default: float,
) -> float:
    value = config.get(
        name,
        default,
    )

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"Promotion config '{name}' must be a finite non-negative number.")

    return float(value)


def evaluate_promotion_guardrails(
    *,
    candidate_metrics: Mapping[str, float],
    champion_metrics: Mapping[str, float],
    config: Mapping[str, Any],
) -> PromotionGuardrailResult:
    """
    Compare candidate and champion on the same validation split.

    The candidate must improve overall RMSE sufficiently, avoid excessive
    RMSE regression in required business segments and avoid excessive
    absolute-bias regression.
    """
    promotion = config.get(
        "promotion",
        {},
    )

    if not isinstance(
        promotion,
        Mapping,
    ):
        raise ValueError("Config must contain a valid 'promotion' section.")

    minimum_relative_improvement = _non_negative_float(
        promotion,
        "minimum_relative_rmse_improvement",
        default=0.0,
    )
    maximum_segment_regression = _non_negative_float(
        promotion,
        "maximum_segment_rmse_regression",
        default=0.0,
    )
    maximum_bias_regression = _non_negative_float(
        promotion,
        "maximum_absolute_bias_regression",
        default=0.0,
    )

    required_segments = promotion.get(
        "required_segments",
        [
            "promo",
            "non_promo",
        ],
    )

    if (
        not isinstance(
            required_segments,
            list,
        )
        or not required_segments
        or not all(isinstance(segment, str) and segment for segment in required_segments)
    ):
        raise ValueError(
            "Promotion config 'required_segments' must be a non-empty list of strings."
        )

    candidate_rmse = _metric(
        candidate_metrics,
        "overall_rmse",
        source="Candidate",
    )
    champion_rmse = _metric(
        champion_metrics,
        "overall_rmse",
        source="Champion",
    )

    if champion_rmse <= 0:
        raise ValueError("Champion overall RMSE must be positive.")

    relative_improvement = (champion_rmse - candidate_rmse) / champion_rmse

    reasons: list[str] = []

    if relative_improvement < minimum_relative_improvement:
        reasons.append(
            "Candidate relative RMSE "
            f"improvement "
            f"{relative_improvement:.4%} is "
            "below the required "
            f"{minimum_relative_improvement:.4%}."
        )

    segment_regressions: dict[
        str,
        float,
    ] = {}

    for segment in required_segments:
        metric_name = f"{segment}_rmse"
        candidate_segment = _metric(
            candidate_metrics,
            metric_name,
            source="Candidate",
        )
        champion_segment = _metric(
            champion_metrics,
            metric_name,
            source="Champion",
        )

        if champion_segment <= 0:
            raise ValueError(f"Champion metric '{metric_name}' must be positive.")

        regression = (candidate_segment - champion_segment) / champion_segment

        segment_regressions[segment] = regression

        if regression > maximum_segment_regression:
            reasons.append(
                f"Candidate segment "
                f"'{segment}' RMSE regression "
                f"{regression:.4%} exceeds "
                "the allowed "
                f"{maximum_segment_regression:.4%}."
            )

    candidate_bias = abs(
        _metric(
            candidate_metrics,
            "overall_bias",
            source="Candidate",
        )
    )
    champion_bias = abs(
        _metric(
            champion_metrics,
            "overall_bias",
            source="Champion",
        )
    )
    bias_regression = candidate_bias - champion_bias

    if bias_regression > maximum_bias_regression:
        reasons.append(
            "Candidate absolute-bias "
            f"regression {bias_regression:.4f} "
            "exceeds the allowed "
            f"{maximum_bias_regression:.4f}."
        )

    return PromotionGuardrailResult(
        approved=not reasons,
        reasons=tuple(reasons),
        relative_rmse_improvement=(relative_improvement),
        segment_rmse_regressions=(segment_regressions),
        absolute_bias_regression=(bias_regression),
    )
