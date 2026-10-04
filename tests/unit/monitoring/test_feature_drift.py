from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mlops_sales_forecasting.monitoring.feature_drift import (
    append_feature_drift_history,
    detect_categorical_drift,
    detect_numeric_drift,
    evaluate_feature_drift,
    run_feature_drift_check,
    summarize_feature_drift,
)


def test_numeric_drift_is_detected() -> None:
    result = detect_numeric_drift(
        pd.Series(range(100)),
        pd.Series(range(1000, 1100)),
        feature_name="Sales",
        minimum_samples=50,
        p_value_threshold=0.01,
        statistic_threshold=0.10,
    )

    assert result["feature"] == "Sales"
    assert result["metric_type"] == "ks"
    assert result["drift_detected"] is True
    assert result["reference_n"] == 100
    assert result["current_n"] == 100


def test_numeric_drift_handles_insufficient_samples() -> None:
    result = detect_numeric_drift(
        pd.Series(
            [
                1.0,
                2.0,
            ]
        ),
        pd.Series(
            [
                3.0,
                4.0,
            ]
        ),
        feature_name="Sales",
        minimum_samples=10,
        p_value_threshold=0.01,
        statistic_threshold=0.10,
    )

    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_samples"


def test_categorical_drift_is_detected() -> None:
    reference = pd.Series(["a"] * 90 + ["b"] * 10)
    current = pd.Series(["a"] * 10 + ["b"] * 90)

    result = detect_categorical_drift(
        reference,
        current,
        feature_name="StoreType",
        minimum_samples=50,
        p_value_threshold=0.01,
    )

    assert result["feature"] == "StoreType"
    assert result["metric_type"] == "chisquare"
    assert result["drift_detected"] is True


def test_categorical_drift_handles_missing_values() -> None:
    reference = pd.Series(
        [
            "a",
            None,
            "a",
            None,
        ]
    )
    current = pd.Series(
        [
            "a",
            None,
            "a",
            None,
        ]
    )

    result = detect_categorical_drift(
        reference,
        current,
        feature_name="StateHoliday",
        minimum_samples=1,
        p_value_threshold=0.01,
    )

    assert result["drift_detected"] is False
    assert result["reference_n"] == 4
    assert result["current_n"] == 4


def test_evaluate_feature_drift_skips_missing_features() -> None:
    reference = pd.DataFrame(
        {
            "Sales": range(100),
            "StoreType": ["a"] * 100,
        }
    )
    current = pd.DataFrame(
        {
            "Sales": range(100),
            "StoreType": ["a"] * 100,
        }
    )

    result = evaluate_feature_drift(
        reference=reference,
        current=current,
        numeric_features=[
            "Sales",
            "missing_numeric",
        ],
        categorical_features=[
            "StoreType",
            "missing_category",
        ],
        minimum_samples=50,
    )

    assert result["feature"].tolist() == [
        "Sales",
        "StoreType",
    ]


def test_append_feature_drift_history(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "monitoring" / "feature_drift_history.parquet"
    observed_at = datetime(
        2026,
        9,
        27,
        4,
        0,
        tzinfo=UTC,
    )
    results = pd.DataFrame(
        [
            {
                "feature": "Sales",
                "feature_type": "numeric",
                "metric_type": "ks",
                "score": 0.2,
                "p_value": 0.001,
                "threshold": 0.1,
                "drift_detected": True,
                "reference_n": 100,
                "current_n": 100,
                "reason": "",
            }
        ]
    )

    first_batch = append_feature_drift_history(
        results,
        history_path=str(history_path),
        observed_at=observed_at,
    )
    append_feature_drift_history(
        results,
        history_path=str(history_path),
        observed_at=observed_at,
    )

    history = pd.read_parquet(history_path)

    assert len(first_batch) == 1
    assert len(history) == 2
    assert history["feature"].tolist() == [
        "Sales",
        "Sales",
    ]


def test_summarize_feature_drift() -> None:
    results = pd.DataFrame(
        {
            "feature": [
                "Sales",
                "Promo",
            ],
            "drift_detected": [
                True,
                False,
            ],
        }
    )

    summary = summarize_feature_drift(results)

    assert summary == {
        "checked_features": 2,
        "drifted_features": 1,
        "drifted_feature_names": [
            "Sales",
        ],
    }


def test_run_feature_drift_check_without_data_is_safe(
    tmp_path: Path,
) -> None:
    result = run_feature_drift_check(
        reference=pd.DataFrame(),
        current=pd.DataFrame(),
        history_path=str(tmp_path / "feature_drift_history.parquet"),
        numeric_features=[
            "Sales",
        ],
        categorical_features=[],
    )

    assert result.empty


def test_categorical_drift_handles_missing_categorical_values() -> None:
    reference = pd.Series(
        pd.Categorical(
            [
                "a",
                "a",
                "b",
                None,
            ]
        )
    )
    current = pd.Series(
        pd.Categorical(
            [
                "a",
                "b",
                "b",
                None,
            ]
        )
    )

    result = detect_categorical_drift(
        reference,
        current,
        feature_name="category",
        minimum_samples=1,
        p_value_threshold=0.05,
    )

    assert result["feature"] == "category"
    assert result["feature_type"] == "categorical"
    assert result["metric_type"] == "chisquare"
    assert result["reference_n"] == 4
    assert result["current_n"] == 4
    assert isinstance(
        result["drift_detected"],
        bool,
    )
