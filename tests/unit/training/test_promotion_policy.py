import pytest

from src.training.promotion_policy import (
    evaluate_promotion_policy,
)


POLICY_CONFIG = {
    "minimum_validation_rows": 1000,
    "minimum_segment_rows": 100,
    "minimum_relative_rmse_improvement": 0.005,
    "maximum_segment_rmse_regression": 0.02,
    "maximum_absolute_bias_regression": 100.0,
    "required_segments": [
        "promo",
        "non_promo",
    ],
}


def test_accepts_candidate_that_passes_all_gates():
    decision = evaluate_promotion_policy(
        candidate_metrics={
            "overall_rmse": 950.0,
            "promo_rmse": 900.0,
            "non_promo_rmse": 980.0,
            "overall_bias": 80.0,
        },
        champion_metrics={
            "overall_rmse": 1000.0,
            "promo_rmse": 1000.0,
            "non_promo_rmse": 1000.0,
            "overall_bias": 100.0,
        },
        segment_rows={
            "promo": 1000,
            "non_promo": 4000,
        },
        validation_rows=5000,
        config=POLICY_CONFIG,
    )

    assert decision.accepted is True
    assert all(
        check.passed
        for check in decision.checks.values()
    )
    assert decision.reasons == []


def test_rejects_candidate_without_minimum_overall_gain():
    decision = evaluate_promotion_policy(
        candidate_metrics={
            "overall_rmse": 998.0,
            "promo_rmse": 950.0,
            "non_promo_rmse": 1000.0,
            "overall_bias": 100.0,
        },
        champion_metrics={
            "overall_rmse": 1000.0,
            "promo_rmse": 1000.0,
            "non_promo_rmse": 1000.0,
            "overall_bias": 100.0,
        },
        segment_rows={
            "promo": 1000,
            "non_promo": 4000,
        },
        validation_rows=5000,
        config=POLICY_CONFIG,
    )

    assert decision.accepted is False
    assert (
        decision.checks[
            "overall_rmse"
        ].passed
        is False
    )


def test_rejects_candidate_with_segment_regression():
    decision = evaluate_promotion_policy(
        candidate_metrics={
            "overall_rmse": 950.0,
            "promo_rmse": 900.0,
            "non_promo_rmse": 1030.0,
            "overall_bias": 100.0,
        },
        champion_metrics={
            "overall_rmse": 1000.0,
            "promo_rmse": 1000.0,
            "non_promo_rmse": 1000.0,
            "overall_bias": 100.0,
        },
        segment_rows={
            "promo": 1000,
            "non_promo": 4000,
        },
        validation_rows=5000,
        config=POLICY_CONFIG,
    )

    assert decision.accepted is False
    assert (
        decision.checks[
            "non_promo_rmse"
        ].passed
        is False
    )


def test_rejects_candidate_with_excessive_bias_regression():
    decision = evaluate_promotion_policy(
        candidate_metrics={
            "overall_rmse": 950.0,
            "promo_rmse": 950.0,
            "non_promo_rmse": 950.0,
            "overall_bias": 250.0,
        },
        champion_metrics={
            "overall_rmse": 1000.0,
            "promo_rmse": 1000.0,
            "non_promo_rmse": 1000.0,
            "overall_bias": 100.0,
        },
        segment_rows={
            "promo": 1000,
            "non_promo": 4000,
        },
        validation_rows=5000,
        config=POLICY_CONFIG,
    )

    assert decision.accepted is False
    assert (
        decision.checks[
            "overall_bias"
        ].passed
        is False
    )


def test_fails_closed_with_too_few_validation_rows():
    with pytest.raises(
        ValueError,
        match="Not enough validation rows",
    ):
        evaluate_promotion_policy(
            candidate_metrics={
                "overall_rmse": 900.0,
                "promo_rmse": 900.0,
                "non_promo_rmse": 900.0,
                "overall_bias": 0.0,
            },
            champion_metrics={
                "overall_rmse": 1000.0,
                "promo_rmse": 1000.0,
                "non_promo_rmse": 1000.0,
                "overall_bias": 0.0,
            },
            segment_rows={
                "promo": 1000,
                "non_promo": 4000,
            },
            validation_rows=100,
            config=POLICY_CONFIG,
        )


def test_fails_closed_when_segment_metric_is_missing():
    with pytest.raises(
        ValueError,
        match="Candidate promotion metrics missing",
    ):
        evaluate_promotion_policy(
            candidate_metrics={
                "overall_rmse": 900.0,
                "promo_rmse": 900.0,
                "overall_bias": 0.0,
            },
            champion_metrics={
                "overall_rmse": 1000.0,
                "promo_rmse": 1000.0,
                "non_promo_rmse": 1000.0,
                "overall_bias": 0.0,
            },
            segment_rows={
                "promo": 1000,
                "non_promo": 4000,
            },
            validation_rows=5000,
            config=POLICY_CONFIG,
        )

def test_fails_closed_with_too_few_segment_rows():
    with pytest.raises(
        ValueError,
        match=(
            "Not enough validation rows "
            "for segment 'promo'"
        ),
    ):
        evaluate_promotion_policy(
            candidate_metrics={
                "overall_rmse": 900.0,
                "promo_rmse": 850.0,
                "non_promo_rmse": 950.0,
                "overall_bias": 0.0,
            },
            champion_metrics={
                "overall_rmse": 1000.0,
                "promo_rmse": 1000.0,
                "non_promo_rmse": 1000.0,
                "overall_bias": 0.0,
            },
            validation_rows=5000,
            segment_rows={
                "promo": 20,
                "non_promo": 4980,
            },
            config=POLICY_CONFIG,
        )