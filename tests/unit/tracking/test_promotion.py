import pytest

from mlops_sales_forecasting.tracking.promotion import (
    MetricDirection,
    PromotionDecision,
    PromotionPolicy,
    evaluate_promotion,
)


def test_maximized_metric_promotes_better_candidate() -> None:
    decision = evaluate_promotion(
        candidate_metrics={
            "roc_auc": 0.87,
        },
        champion_metrics={
            "roc_auc": 0.84,
        },
        policy=PromotionPolicy(
            metric_name="roc_auc",
            direction=MetricDirection.MAXIMIZE,
            minimum_improvement=0.02,
        ),
    )

    assert decision == PromotionDecision(
        promote=True,
        metric_name="roc_auc",
        candidate_value=0.87,
        champion_value=0.84,
        improvement=pytest.approx(0.03),
        reason=(
            "Candidate satisfies the configured "
            "promotion threshold."
        ),
    )


def test_maximized_metric_rejects_worse_candidate() -> None:
    decision = evaluate_promotion(
        candidate_metrics={
            "roc_auc": 0.82,
        },
        champion_metrics={
            "roc_auc": 0.84,
        },
        policy=PromotionPolicy(
            metric_name="roc_auc",
            direction=MetricDirection.MAXIMIZE,
        ),
    )

    assert decision.promote is False
    assert decision.improvement == pytest.approx(
        -0.02
    )


def test_minimized_metric_promotes_lower_value() -> None:
    decision = evaluate_promotion(
        candidate_metrics={
            "rmse": 720.0,
        },
        champion_metrics={
            "rmse": 750.0,
        },
        policy=PromotionPolicy(
            metric_name="rmse",
            direction=MetricDirection.MINIMIZE,
            minimum_improvement=20.0,
        ),
    )

    assert decision.promote is True
    assert decision.improvement == 30.0


def test_minimized_metric_rejects_higher_value() -> None:
    decision = evaluate_promotion(
        candidate_metrics={
            "rmse": 780.0,
        },
        champion_metrics={
            "rmse": 750.0,
        },
        policy=PromotionPolicy(
            metric_name="rmse",
            direction=MetricDirection.MINIMIZE,
        ),
    )

    assert decision.promote is False
    assert decision.improvement == -30.0


def test_improvement_may_equal_threshold() -> None:
    decision = evaluate_promotion(
        candidate_metrics={
            "score": 0.85,
        },
        champion_metrics={
            "score": 0.80,
        },
        policy=PromotionPolicy(
            metric_name="score",
            direction=MetricDirection.MAXIMIZE,
            minimum_improvement=0.05,
        ),
    )

    assert decision.promote is True


def test_candidate_can_be_initial_champion() -> None:
    decision = evaluate_promotion(
        candidate_metrics={
            "score": 0.8,
        },
        champion_metrics=None,
        policy=PromotionPolicy(
            metric_name="score",
            direction=MetricDirection.MAXIMIZE,
        ),
    )

    assert decision.promote is True
    assert decision.champion_value is None
    assert decision.improvement is None


def test_initial_promotion_can_be_disabled() -> None:
    decision = evaluate_promotion(
        candidate_metrics={
            "score": 0.8,
        },
        champion_metrics=None,
        policy=PromotionPolicy(
            metric_name="score",
            direction=MetricDirection.MAXIMIZE,
            allow_initial_champion=False,
        ),
    )

    assert decision.promote is False
    assert "disabled" in decision.reason


@pytest.mark.parametrize(
    "source",
    [
        "candidate",
        "champion",
    ],
)
def test_required_metric_must_exist(
    source: str,
) -> None:
    candidate_metrics = {
        "score": 0.8,
    }
    champion_metrics = {
        "score": 0.7,
    }

    if source == "candidate":
        candidate_metrics = {}
    else:
        champion_metrics = {}

    with pytest.raises(
        ValueError,
        match="promotion metric 'score'",
    ):
        evaluate_promotion(
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            policy=PromotionPolicy(
                metric_name="score",
                direction=(
                    MetricDirection.MAXIMIZE
                ),
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        True,
        "invalid",
    ],
)
def test_promotion_metric_must_be_finite(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        evaluate_promotion(
            candidate_metrics={
                "score": value,
            },  # type: ignore[dict-item]
            champion_metrics=None,
            policy=PromotionPolicy(
                metric_name="score",
                direction=(
                    MetricDirection.MAXIMIZE
                ),
            ),
        )


@pytest.mark.parametrize(
    "minimum_improvement",
    [
        -0.1,
        float("nan"),
        float("inf"),
        True,
        "invalid",
    ],
)
def test_minimum_improvement_must_be_valid(
    minimum_improvement: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Minimum improvement",
    ):
        PromotionPolicy(
            metric_name="score",
            direction=MetricDirection.MAXIMIZE,
            minimum_improvement=minimum_improvement,
        )  # type: ignore[arg-type]


def test_policy_requires_metric_name() -> None:
    with pytest.raises(
        ValueError,
        match="metric name",
    ):
        PromotionPolicy(
            metric_name="",
            direction=MetricDirection.MAXIMIZE,
        )


def test_policy_requires_metric_direction() -> None:
    with pytest.raises(
        TypeError,
        match="direction",
    ):
        PromotionPolicy(
            metric_name="score",
            direction="maximize",
        )  # type: ignore[arg-type]


def test_evaluation_requires_policy() -> None:
    with pytest.raises(
        TypeError,
        match="requires PromotionPolicy",
    ):
        evaluate_promotion(
            candidate_metrics={
                "score": 0.8,
            },
            champion_metrics=None,
            policy=object(),  # type: ignore[arg-type]
        )