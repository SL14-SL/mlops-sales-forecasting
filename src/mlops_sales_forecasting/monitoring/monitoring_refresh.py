import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd

from ..configs.paths import join_uri
from ..storage.filesystem import (
    ensure_dir,
    file_exists,
    list_files,
)
from .feature_drift import run_feature_drift_check
from .performance import (
    compute_rolling_metrics,
    prepare_evaluation_frame,
    save_metrics,
)


@dataclass(frozen=True)
class MonitoringRefreshResult:
    """Outcome of refreshing persisted monitoring evidence."""

    ground_truth_rows: int
    inference_rows: int
    performance_updated: bool
    performance_rows: int
    feature_drift_updated: bool
    feature_drift_rows: int
    performance_reason: str


def _require_mapping(
    mapping: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = mapping.get(name)

    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a valid '{name}' section.")

    return value


def _require_path(
    paths: Mapping[str, Any],
    name: str,
) -> str:
    value = paths.get(name)

    if not isinstance(value, str) or not value:
        raise ValueError(f"Config must contain a non-empty 'paths.{name}' value.")

    return value


def rebuild_cumulative_ground_truth(
    batch_files: list[str],
    *,
    output_path: str,
) -> pd.DataFrame:
    """Rebuild deduplicated Ground Truth from all batches."""
    if not batch_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for batch_path in batch_files:
        with fsspec.open(
            batch_path,
            "rb",
        ) as file:
            content = file.read()

        frame = pd.read_csv(
            io.BytesIO(content),
            parse_dates=[
                "Date",
            ],
            dtype={
                "StateHoliday": str,
            },
        )
        frames.append(frame)

    cumulative = pd.concat(
        frames,
        ignore_index=True,
    )

    required_columns = {
        "Store",
        "Date",
        "Sales",
    }
    missing_columns = required_columns - set(cumulative.columns)

    if missing_columns:
        raise ValueError(f"Ground Truth is missing columns: {sorted(missing_columns)}.")

    cumulative["Date"] = pd.to_datetime(
        cumulative["Date"],
        errors="coerce",
    )
    cumulative["Store"] = pd.to_numeric(
        cumulative["Store"],
        errors="coerce",
    ).astype("Int64")
    cumulative["Sales"] = pd.to_numeric(
        cumulative["Sales"],
        errors="coerce",
    )

    cumulative = (
        cumulative.dropna(
            subset=[
                "Store",
                "Date",
                "Sales",
            ]
        )
        .sort_values(
            [
                "Date",
                "Store",
            ]
        )
        .drop_duplicates(
            subset=[
                "Store",
                "Date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if not output_path.startswith("gs://"):
        ensure_dir(str(Path(output_path).parent))

    with fsspec.open(
        output_path,
        "w",
    ) as file:
        cumulative.to_csv(
            file,
            index=False,
        )

    return cumulative


def load_inference_history(
    inference_files: list[str],
) -> pd.DataFrame:
    """Load all immutable inference partitions."""
    if not inference_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for inference_path in inference_files:
        with fsspec.open(
            inference_path,
            "rb",
        ) as file:
            frames.append(pd.read_parquet(file))

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    required_columns = {
        "Store",
        "Date",
        "prediction",
        "request_id",
        "row_index",
        "timestamp",
    }
    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(f"Inference history is missing columns: {sorted(missing_columns)}.")

    history["Store"] = pd.to_numeric(
        history["Store"],
        errors="coerce",
    ).astype("Int64")
    history["Date"] = pd.to_datetime(
        history["Date"],
        errors="coerce",
    )
    history["prediction"] = pd.to_numeric(
        history["prediction"],
        errors="coerce",
    )
    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
        utc=True,
    )

    history = (
        history.dropna(
            subset=[
                "Store",
                "Date",
                "prediction",
                "timestamp",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=[
                "request_id",
                "row_index",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return history


def _latest_predictions_per_key(
    inference_history: pd.DataFrame,
) -> pd.DataFrame:
    """Select the latest prediction for each Store-Date key."""
    if inference_history.empty:
        return inference_history.copy()

    return (
        inference_history.sort_values("timestamp")
        .drop_duplicates(
            subset=[
                "Store",
                "Date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def _recent_inference_window(
    inference_history: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    """Select the latest configured inference period."""
    if inference_history.empty:
        return inference_history.copy()

    if lookback_days < 1:
        raise ValueError("Feature-drift lookback must be positive.")

    latest_timestamp = inference_history["timestamp"].max()
    cutoff = latest_timestamp - pd.to_timedelta(
        lookback_days,
        unit="d",
    )

    return inference_history.loc[inference_history["timestamp"] >= cutoff].copy()


def _load_reference_features(
    path: str,
) -> pd.DataFrame:
    """Load feature reference data when available."""
    if not file_exists(path):
        return pd.DataFrame()

    with fsspec.open(
        path,
        "rb",
    ) as file:
        return pd.read_parquet(file)


def refresh_monitoring_signals(
    *,
    config: Mapping[str, Any],
) -> MonitoringRefreshResult:
    """Refresh performance and feature-drift evidence."""
    paths = _require_mapping(
        config,
        "paths",
    )
    monitoring = _require_mapping(
        config,
        "monitoring",
    )
    drift_settings = _require_mapping(
        monitoring,
        "feature_drift",
    )
    retraining_settings = _require_mapping(
        monitoring,
        "retraining",
    )

    raw_path = _require_path(
        paths,
        "raw_data",
    )
    predictions_path = _require_path(
        paths,
        "predictions",
    )
    monitoring_path = _require_path(
        paths,
        "monitoring",
    )
    features_path = _require_path(
        paths,
        "features",
    )

    ground_truth_files = list_files(
        join_uri(
            raw_path,
            "new_batches",
            "ground_truth_*.csv",
        )
    )
    inference_files = list_files(
        join_uri(
            predictions_path,
            "history",
            "date=*",
            "*.parquet",
        )
    )

    cumulative_path = join_uri(
        monitoring_path,
        "cumulative_ground_truth.csv",
    )
    performance_path = join_uri(
        monitoring_path,
        "performance_rolling.parquet",
    )
    drift_history_path = join_uri(
        monitoring_path,
        "feature_drift_history.parquet",
    )
    reference_path = join_uri(
        features_path,
        "features.parquet",
    )

    ground_truth = rebuild_cumulative_ground_truth(
        ground_truth_files,
        output_path=cumulative_path,
    )
    inference_history = load_inference_history(inference_files)

    performance_updated = False
    performance_rows = 0
    performance_reason = "No Ground-Truth batches available."

    if not ground_truth.empty and not inference_history.empty:
        latest_predictions = _latest_predictions_per_key(inference_history)

        try:
            performance_settings = _require_mapping(
                retraining_settings,
                "performance",
            )

            evaluation_ground_truth = ground_truth

            if bool(
                performance_settings.get(
                    "open_store_only",
                    False,
                )
            ):
                if "Open" not in evaluation_ground_truth.columns:
                    raise ValueError(
                        "Open-store-only performance "
                        "evaluation requires the "
                        "Ground-Truth column 'Open'."
                    )

                open_values = pd.to_numeric(
                    evaluation_ground_truth["Open"],
                    errors="coerce",
                )

                evaluation_ground_truth = evaluation_ground_truth.loc[open_values.eq(1)].copy()

                if evaluation_ground_truth.empty:
                    raise ValueError(
                        "No open-store Ground-Truth rows are available for performance evaluation."
                    )

            joined = prepare_evaluation_frame(
                predictions=latest_predictions,
                ground_truth=(evaluation_ground_truth),
                join_columns=(
                    "Store",
                    "Date",
                ),
                actual_column="Sales",
                prediction_column="prediction",
                time_column="Date",
            )

            minimum_samples = int(
                performance_settings.get(
                    "minimum_samples",
                    retraining_settings.get(
                        "minimum_new_training_rows",
                        500,
                    ),
                )
            )
            rolling_window = str(
                performance_settings.get(
                    "rolling_window",
                    "7D",
                )
            )

            metrics = compute_rolling_metrics(
                joined,
                time_column="Date",
                window=rolling_window,
                actual_column="Sales",
                prediction_column="prediction",
                minimum_samples=minimum_samples,
            )

            if metrics.empty:
                performance_reason = "Not enough matched samples for a performance window."
            else:
                save_metrics(
                    metrics,
                    performance_path,
                )
                performance_updated = True
                performance_rows = len(metrics)
                performance_reason = "Performance history refreshed."

        except ValueError as error:
            performance_reason = str(error)

    elif not ground_truth.empty and inference_history.empty:
        performance_reason = "No inference history available."

    reference_features = _load_reference_features(reference_path)
    current_features = _recent_inference_window(
        inference_history,
        lookback_days=int(
            drift_settings.get(
                "lookback_days",
                14,
            )
        ),
    )

    drift_result = run_feature_drift_check(
        reference=reference_features,
        current=current_features,
        history_path=drift_history_path,
        numeric_features=list(
            drift_settings.get(
                "numeric_features",
                [],
            )
        ),
        categorical_features=list(
            drift_settings.get(
                "categorical_features",
                [],
            )
        ),
        minimum_samples=int(
            drift_settings.get(
                "minimum_samples",
                50,
            )
        ),
        p_value_threshold=float(
            drift_settings.get(
                "p_value_threshold",
                0.01,
            )
        ),
        statistic_threshold=float(
            drift_settings.get(
                "statistic_threshold",
                0.10,
            )
        ),
    )

    return MonitoringRefreshResult(
        ground_truth_rows=len(ground_truth),
        inference_rows=len(inference_history),
        performance_updated=(performance_updated),
        performance_rows=performance_rows,
        feature_drift_updated=(not drift_result.empty),
        feature_drift_rows=len(drift_result),
        performance_reason=performance_reason,
    )
