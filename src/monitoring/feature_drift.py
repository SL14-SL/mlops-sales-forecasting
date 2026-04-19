from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import chisquare, ks_2samp

from src.configs.loader import file_exists, get_path
from src.monitoring.config import get_feature_drift_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MONITORING_PATH = get_path("monitoring")
PREDICTIONS_PATH = get_path("predictions")
VALIDATED_PATH = get_path("validated_data")


def _history_path() -> str:
    return f"{MONITORING_PATH}/feature_drift_history.parquet"


def load_reference_features() -> pd.DataFrame:
    """
    Loads the reference dataset used as baseline for drift detection.
    For v1 we use the validated training data snapshot.
    """
    path = f"{VALIDATED_PATH}/train.parquet"

    if not file_exists(path):
        logger.info(
            "No reference feature data found yet at %s. "
            "Skipping feature drift check during bootstrap run.",
            path,
        )
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    return df


def load_current_inference_features() -> pd.DataFrame:
    """
    Loads the current inference log as 'live' data for drift detection.
    """
    path = f"{PREDICTIONS_PATH}/inference_log.parquet"

    if not file_exists(path):
        logger.info("No inference log found yet.")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _safe_categorical(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna("MISSING")


def detect_numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature_name: str,
    min_samples: int,
    p_value_threshold: float,
    stat_threshold: float,
) -> dict:
    ref = _safe_numeric(reference)
    cur = _safe_numeric(current)

    if len(ref) < min_samples or len(cur) < min_samples:
        return {
            "feature": feature_name,
            "feature_type": "numeric",
            "metric_type": "ks",
            "score": 0.0,
            "p_value": 1.0,
            "threshold": stat_threshold,
            "drift_detected": False,
            "reference_n": int(len(ref)),
            "current_n": int(len(cur)),
            "reason": "insufficient_samples",
        }

    stat, p_value = ks_2samp(ref, cur)
    drift_detected = (p_value < p_value_threshold) and (stat > stat_threshold)

    return {
        "feature": feature_name,
        "feature_type": "numeric",
        "metric_type": "ks",
        "score": float(stat),
        "p_value": float(p_value),
        "threshold": float(stat_threshold),
        "drift_detected": bool(drift_detected),
        "reference_n": int(len(ref)),
        "current_n": int(len(cur)),
        "reason": "",
    }


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature_name: str,
    min_samples: int,
    p_value_threshold: float,
) -> dict:
    ref = _safe_categorical(reference)
    cur = _safe_categorical(current)

    ref_counts = ref.value_counts(dropna=False)
    cur_counts = cur.value_counts(dropna=False)

    ref_n = int(ref_counts.sum())
    cur_n = int(cur_counts.sum())

    if ref_n < min_samples or cur_n < min_samples:
        return {
            "feature": feature_name,
            "feature_type": "categorical",
            "metric_type": "chisquare",
            "score": 0.0,
            "p_value": 1.0,
            "threshold": p_value_threshold,
            "drift_detected": False,
            "reference_n": ref_n,
            "current_n": cur_n,
            "reason": "insufficient_samples",
        }

    categories = sorted(set(ref_counts.index).union(set(cur_counts.index)))
    ref_counts = ref_counts.reindex(categories, fill_value=0)
    cur_counts = cur_counts.reindex(categories, fill_value=0)

    expected = (ref_counts / ref_counts.sum()) * cur_counts.sum()
    expected = expected.clip(lower=1e-9)

    stat, p_value = chisquare(f_obs=cur_counts, f_exp=expected)
    drift_detected = p_value < p_value_threshold

    return {
        "feature": feature_name,
        "feature_type": "categorical",
        "metric_type": "chisquare",
        "score": float(stat),
        "p_value": float(p_value),
        "threshold": float(p_value_threshold),
        "drift_detected": bool(drift_detected),
        "reference_n": ref_n,
        "current_n": cur_n,
        "reason": "",
    }


def append_feature_drift_history(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    output_path = _history_path()
    now = datetime.now(timezone.utc)

    batch_df = pd.DataFrame(results)
    batch_df["timestamp"] = now

    if file_exists(output_path):
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, batch_df], ignore_index=True)
    else:
        if not output_path.startswith("gs://"):
            os.makedirs(MONITORING_PATH, exist_ok=True)
        combined = batch_df

    combined.to_parquet(output_path, index=False)
    return batch_df


def summarize_feature_drift(results_df: pd.DataFrame) -> dict:
    if results_df.empty:
        return {
            "checked_features": 0,
            "drifted_features": 0,
            "drifted_feature_names": [],
        }

    drifted = results_df[results_df["drift_detected"]]

    return {
        "checked_features": int(len(results_df)),
        "drifted_features": int(len(drifted)),
        "drifted_feature_names": drifted["feature"].tolist(),
    }


def run_feature_drift_check() -> pd.DataFrame:
    cfg = get_feature_drift_settings()

    if not cfg.get("enabled", True):
        logger.info("Feature drift monitoring is disabled in monitoring.yaml.")
        return pd.DataFrame()

    numeric_features = cfg.get("numeric_features", [])
    categorical_features = cfg.get("categorical_features", [])
    min_samples = int(cfg.get("min_samples", 50))
    p_value_threshold = float(cfg.get("p_value_threshold", 0.01))
    stat_threshold = float(cfg.get("stat_threshold", 0.10))

    reference_df = load_reference_features()
    current_df = load_current_inference_features()

    if current_df.empty:
        logger.info("Skipping feature drift check because no inference data is available.")
        return pd.DataFrame()

    results: list[dict] = []

    for feature in numeric_features:
        if feature not in reference_df.columns:
            logger.warning(f"Numeric reference feature missing: {feature}")
            continue
        if feature not in current_df.columns:
            logger.warning(f"Numeric current feature missing: {feature}")
            continue

        result = detect_numeric_drift(
            reference_df[feature],
            current_df[feature],
            feature_name=feature,
            min_samples=min_samples,
            p_value_threshold=p_value_threshold,
            stat_threshold=stat_threshold,
        )
        results.append(result)

    for feature in categorical_features:
        if feature not in reference_df.columns:
            logger.warning(f"Categorical reference feature missing: {feature}")
            continue
        if feature not in current_df.columns:
            logger.warning(f"Categorical current feature missing: {feature}")
            continue

        result = detect_categorical_drift(
            reference_df[feature],
            current_df[feature],
            feature_name=feature,
            min_samples=min_samples,
            p_value_threshold=p_value_threshold,
        )
        results.append(result)

    latest_df = append_feature_drift_history(results)

    summary = summarize_feature_drift(latest_df)
    logger.info(
        "Feature drift check finished | "
        f"checked_features={summary['checked_features']} | "
        f"drifted_features={summary['drifted_features']} | "
        f"drifted_feature_names={summary['drifted_feature_names']}"
    )

    return latest_df