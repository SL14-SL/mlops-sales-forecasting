from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import chisquare, ks_2samp

from ..storage.filesystem import (
    ensure_dir,
    file_exists,
)


def _numeric_values(
    series: pd.Series,
) -> pd.Series:
    """Return finite numeric values from a feature series."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    )
    return values.loc[values.notna()]


def _categorical_values(
    series: pd.Series,
) -> pd.Series:
    """Return normalized categorical values."""
    return series.fillna("MISSING").astype(str)


def _insufficient_samples_result(
    *,
    feature_name: str,
    feature_type: str,
    metric_type: str,
    threshold: float,
    reference_count: int,
    current_count: int,
) -> dict[str, Any]:
    """Build a result for a feature without enough samples."""
    return {
        "feature": feature_name,
        "feature_type": feature_type,
        "metric_type": metric_type,
        "score": 0.0,
        "p_value": 1.0,
        "threshold": float(threshold),
        "drift_detected": False,
        "reference_n": reference_count,
        "current_n": current_count,
        "reason": "insufficient_samples",
    }


def detect_numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature_name: str,
    minimum_samples: int,
    p_value_threshold: float,
    statistic_threshold: float,
) -> dict[str, Any]:
    """Evaluate one numeric feature using a KS test."""
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive.")

    if not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between zero and one.")

    if not 0 <= statistic_threshold <= 1:
        raise ValueError("statistic_threshold must be between zero and one.")

    reference_values = _numeric_values(reference)
    current_values = _numeric_values(current)

    if len(reference_values) < minimum_samples or len(current_values) < minimum_samples:
        return _insufficient_samples_result(
            feature_name=feature_name,
            feature_type="numeric",
            metric_type="ks",
            threshold=statistic_threshold,
            reference_count=len(reference_values),
            current_count=len(current_values),
        )

    statistic, p_value = ks_2samp(
        reference_values,
        current_values,
    )
    drift_detected = p_value < p_value_threshold and statistic > statistic_threshold

    return {
        "feature": feature_name,
        "feature_type": "numeric",
        "metric_type": "ks",
        "score": float(statistic),
        "p_value": float(p_value),
        "threshold": float(statistic_threshold),
        "drift_detected": bool(drift_detected),
        "reference_n": int(len(reference_values)),
        "current_n": int(len(current_values)),
        "reason": "",
    }


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature_name: str,
    minimum_samples: int,
    p_value_threshold: float,
) -> dict[str, Any]:
    """Evaluate one categorical feature using a chi-square test."""
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive.")

    if not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between zero and one.")

    reference_values = _categorical_values(reference)
    current_values = _categorical_values(current)

    reference_count = len(reference_values)
    current_count = len(current_values)

    if reference_count < minimum_samples or current_count < minimum_samples:
        return _insufficient_samples_result(
            feature_name=feature_name,
            feature_type="categorical",
            metric_type="chisquare",
            threshold=p_value_threshold,
            reference_count=reference_count,
            current_count=current_count,
        )

    categories = sorted(set(reference_values) | set(current_values))

    reference_counts = (
        reference_values.value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
        .astype(float)
    )
    current_counts = (
        current_values.value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
        .astype(float)
    )

    # Laplace smoothing handles categories that occur only
    # in the current or reference distribution.
    reference_counts += 1.0
    current_counts += 1.0

    expected_counts = reference_counts / reference_counts.sum() * current_counts.sum()

    statistic, p_value = chisquare(
        f_obs=current_counts,
        f_exp=expected_counts,
    )

    return {
        "feature": feature_name,
        "feature_type": "categorical",
        "metric_type": "chisquare",
        "score": float(statistic),
        "p_value": float(p_value),
        "threshold": float(p_value_threshold),
        "drift_detected": bool(p_value < p_value_threshold),
        "reference_n": int(reference_count),
        "current_n": int(current_count),
        "reason": "",
    }


def evaluate_feature_drift(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    minimum_samples: int = 50,
    p_value_threshold: float = 0.01,
    statistic_threshold: float = 0.10,
) -> pd.DataFrame:
    """Evaluate configured features shared by both datasets."""
    results: list[dict[str, Any]] = []

    for feature in numeric_features:
        if feature not in reference.columns or feature not in current.columns:
            continue

        results.append(
            detect_numeric_drift(
                reference[feature],
                current[feature],
                feature_name=feature,
                minimum_samples=minimum_samples,
                p_value_threshold=p_value_threshold,
                statistic_threshold=statistic_threshold,
            )
        )

    for feature in categorical_features:
        if feature not in reference.columns or feature not in current.columns:
            continue

        results.append(
            detect_categorical_drift(
                reference[feature],
                current[feature],
                feature_name=feature,
                minimum_samples=minimum_samples,
                p_value_threshold=p_value_threshold,
            )
        )

    return pd.DataFrame(results)


def append_feature_drift_history(
    results: pd.DataFrame,
    *,
    history_path: str,
    observed_at: datetime | None = None,
) -> pd.DataFrame:
    """Append one drift evaluation to persistent history."""
    if results.empty:
        return results.copy()

    batch = results.copy()
    batch["timestamp"] = observed_at or datetime.now(UTC)

    if file_exists(history_path):
        existing = pd.read_parquet(history_path)
        combined = pd.concat(
            [
                existing,
                batch,
            ],
            ignore_index=True,
        )
    else:
        if not history_path.startswith("gs://"):
            ensure_dir(str(Path(history_path).parent))

        combined = batch

    combined.to_parquet(
        history_path,
        index=False,
    )
    return batch


def summarize_feature_drift(
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize feature-level drift decisions."""
    if results.empty:
        return {
            "checked_features": 0,
            "drifted_features": 0,
            "drifted_feature_names": [],
        }

    if "drift_detected" not in results.columns:
        raise KeyError("Drift results have no 'drift_detected' column.")

    drifted = results.loc[results["drift_detected"].astype(bool)]

    return {
        "checked_features": int(len(results)),
        "drifted_features": int(len(drifted)),
        "drifted_feature_names": (drifted["feature"].tolist()),
    }


def run_feature_drift_check(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    history_path: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    minimum_samples: int = 50,
    p_value_threshold: float = 0.01,
    statistic_threshold: float = 0.10,
    observed_at: datetime | None = None,
) -> pd.DataFrame:
    """Evaluate features and append the result to history."""
    if reference.empty or current.empty:
        return pd.DataFrame()

    results = evaluate_feature_drift(
        reference=reference,
        current=current,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        minimum_samples=minimum_samples,
        p_value_threshold=p_value_threshold,
        statistic_threshold=statistic_threshold,
    )

    return append_feature_drift_history(
        results,
        history_path=history_path,
        observed_at=observed_at,
    )
