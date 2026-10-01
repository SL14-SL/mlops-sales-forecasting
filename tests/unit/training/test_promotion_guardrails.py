import pytest

from mlops_sales_forecasting.training.promotion_guardrails import (
    evaluate_promotion_guardrails,
)


def config() -> dict:
    return {
        "promotion": {
            "minimum_relative_rmse_improvement": 0.005,
            "maximum_segment_rmse_regression": 0.02,
            "maximum_absolute_bias_regression": 100.0,
            "required_segments": [
                "promo",
                "non_promo",
            ],
        },
    }


def test_approves_balanced_candidate() -> None:
    result = evaluate_promotion_guardrails(
        candidate_metrics={
            "overall_rmse": 900.0,
            "promo_rmse": 1100.0,
            "non_promo_rmse": 700.0,
            "overall_bias": 40.0,
        },
        champion_metrics={
            "overall_rmse": 1000.0,
            "promo_rmse": 1300.0,
            "non_promo_rmse": 710.0,
            "overall_bias": 50.0,
        },
        config=config(),
    )

    assert result.approved is True
    assert result.reasons == ()
    assert result.relative_rmse_improvement == pytest.approx(0.10)
    assert result.segment_rmse_regressions["non_promo"] == pytest.approx(-10.0 / 710.0)
    assert result.absolute_bias_regression == pytest.approx(-10.0)


def test_rejects_segment_regression() -> None:
    result = evaluate_promotion_guardrails(
        candidate_metrics={
            "overall_rmse": 1024.7679757652113,
            "promo_rmse": 1334.780099831068,
            "non_promo_rmse": 802.3508572819574,
            "overall_bias": 119.31365785125469,
        },
        champion_metrics={
            "overall_rmse": 1135.8445563089435,
            "promo_rmse": 1617.456074278585,
            "non_promo_rmse": 743.9518521046759,
            "overall_bias": 100.0,
        },
        config=config(),
    )

    assert result.approved is False
    assert result.relative_rmse_improvement == pytest.approx(
        (1135.8445563089435 - 1024.7679757652113) / 1135.8445563089435
    )
    assert result.segment_rmse_regressions["non_promo"] > 0.02
    assert any("non_promo" in reason for reason in result.reasons)


def test_rejects_insufficient_overall_improvement() -> None:
    result = evaluate_promotion_guardrails(
        candidate_metrics={
            "overall_rmse": 999.0,
            "promo_rmse": 1000.0,
            "non_promo_rmse": 900.0,
            "overall_bias": 50.0,
        },
        champion_metrics={
            "overall_rmse": 1000.0,
            "promo_rmse": 1000.0,
            "non_promo_rmse": 900.0,
            "overall_bias": 50.0,
        },
        config=config(),
    )

    assert result.approved is False
    assert any("relative RMSE improvement" in reason for reason in result.reasons)


def test_rejects_absolute_bias_regression() -> None:
    result = evaluate_promotion_guardrails(
        candidate_metrics={
            "overall_rmse": 900.0,
            "promo_rmse": 1000.0,
            "non_promo_rmse": 800.0,
            "overall_bias": -180.0,
        },
        champion_metrics={
            "overall_rmse": 1000.0,
            "promo_rmse": 1100.0,
            "non_promo_rmse": 850.0,
            "overall_bias": 50.0,
        },
        config=config(),
    )

    assert result.approved is False
    assert result.absolute_bias_regression == pytest.approx(130.0)
    assert any("absolute-bias" in reason for reason in result.reasons)


def test_missing_metric_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="non_promo_rmse",
    ):
        evaluate_promotion_guardrails(
            candidate_metrics={
                "overall_rmse": 900.0,
                "promo_rmse": 1000.0,
                "overall_bias": 0.0,
            },
            champion_metrics={
                "overall_rmse": 1000.0,
                "promo_rmse": 1100.0,
                "non_promo_rmse": 850.0,
                "overall_bias": 0.0,
            },
            config=config(),
        )
