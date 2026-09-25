from collections.abc import Mapping
from typing import Any

from .promotion import (
    MetricDirection,
    PromotionPolicy,
)


def _promotion_section(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the configured promotion section."""

    tracking = config.get("tracking")

    if not isinstance(tracking, Mapping):
        raise ValueError(
            "Config must contain a valid "
            "'tracking' section."
        )

    promotion = tracking.get("promotion")

    if not isinstance(promotion, Mapping):
        raise ValueError(
            "Tracking config must contain a valid "
            "'promotion' section."
        )

    return promotion


def _required_string(
    section: Mapping[str, Any],
    name: str,
) -> str:
    """Return one required non-empty string value."""

    value = section.get(name)

    if (
        not isinstance(value, str)
        or not value.strip()
        or value.startswith("${")
    ):
        raise ValueError(
            f"Promotion config '{name}' must be "
            "a resolved non-empty string."
        )

    return value


def load_promotion_policy(
    config: Mapping[str, Any],
) -> PromotionPolicy:
    """Build the model-promotion policy from configuration."""

    section = _promotion_section(config)

    metric_name = _required_string(
        section,
        "metric_name",
    )
    direction_value = _required_string(
        section,
        "direction",
    )

    try:
        direction = MetricDirection(
            direction_value
        )
    except ValueError as error:
        valid_directions = ", ".join(
            direction.value
            for direction in MetricDirection
        )
        raise ValueError(
            "Promotion config 'direction' must be "
            f"one of: {valid_directions}."
        ) from error

    minimum_improvement = section.get(
        "minimum_improvement",
        0.0,
    )
    allow_initial_champion = section.get(
        "allow_initial_champion",
        True,
    )

    return PromotionPolicy(
        metric_name=metric_name,
        direction=direction,
        minimum_improvement=minimum_improvement,
        allow_initial_champion=allow_initial_champion,
    )