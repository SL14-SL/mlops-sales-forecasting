import pytest

from mlops_sales_forecasting.tracking.promotion import (
    MetricDirection,
)
from mlops_sales_forecasting.tracking.promotion_config import (
    load_promotion_policy,
)


def build_config() -> dict:
    return {
        "tracking": {
            "promotion": {
                "metric_name": "score",
                "direction": "maximize",
                "minimum_improvement": 0.05,
                "allow_initial_champion": True,
            }
        }
    }


def test_loads_promotion_policy() -> None:
    policy = load_promotion_policy(
        build_config()
    )

    assert policy.metric_name == "score"
    assert (
        policy.direction
        is MetricDirection.MAXIMIZE
    )
    assert policy.minimum_improvement == 0.05
    assert policy.allow_initial_champion is True


def test_uses_optional_defaults() -> None:
    config = build_config()
    promotion = config["tracking"]["promotion"]
    del promotion["minimum_improvement"]
    del promotion["allow_initial_champion"]

    policy = load_promotion_policy(config)

    assert policy.minimum_improvement == 0.0
    assert policy.allow_initial_champion is True


def test_requires_tracking_section() -> None:
    with pytest.raises(
        ValueError,
        match="valid 'tracking' section",
    ):
        load_promotion_policy({})


def test_requires_promotion_section() -> None:
    with pytest.raises(
        ValueError,
        match="valid 'promotion' section",
    ):
        load_promotion_policy(
            {"tracking": {}}
        )


@pytest.mark.parametrize(
    "metric_name",
    [
        "",
        "   ",
        "${PROMOTION_METRIC}",
        None,
    ],
)
def test_rejects_invalid_metric_name(
    metric_name: object,
) -> None:
    config = build_config()
    config["tracking"]["promotion"][
        "metric_name"
    ] = metric_name

    with pytest.raises(
        ValueError,
        match="metric_name",
    ):
        load_promotion_policy(config)


def test_rejects_unknown_direction() -> None:
    config = build_config()
    config["tracking"]["promotion"][
        "direction"
    ] = "sideways"

    with pytest.raises(
        ValueError,
        match="maximize, minimize",
    ):
        load_promotion_policy(config)


def test_supports_minimize_direction() -> None:
    config = build_config()
    config["tracking"]["promotion"][
        "direction"
    ] = "minimize"

    policy = load_promotion_policy(config)

    assert (
        policy.direction
        is MetricDirection.MINIMIZE
    )


def test_rejects_negative_minimum_improvement() -> None:
    config = build_config()
    config["tracking"]["promotion"][
        "minimum_improvement"
    ] = -0.01

    with pytest.raises(
        ValueError,
        match="finite non-negative",
    ):
        load_promotion_policy(config)


def test_rejects_non_boolean_initial_setting() -> None:
    config = build_config()
    config["tracking"]["promotion"][
        "allow_initial_champion"
    ] = "yes"

    with pytest.raises(
        TypeError,
        match="must be a boolean",
    ):
        load_promotion_policy(config)