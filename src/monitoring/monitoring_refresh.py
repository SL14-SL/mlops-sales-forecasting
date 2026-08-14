from __future__ import annotations

import io
from dataclasses import dataclass

import fsspec
import pandas as pd

from src.configs.loader import (
    file_exists,
    get_path,
    join_uri,
)
from src.monitoring.feature_drift import (
    run_feature_drift_check,
)
from src.monitoring.performance import (
    compute_rolling_metrics,
    prepare_joined_evaluation_frame,
    save_metrics,
)


@dataclass(frozen=True)
class MonitoringRefreshResult:
    ground_truth_rows: int
    performance_updated: bool
    performance_rows: int
    feature_drift_updated: bool
    feature_drift_rows: int
    performance_reason: str


def _list_files(pattern: str) -> list[str]:
    filesystem, fs_pattern = (
        fsspec.core.url_to_fs(pattern)
    )

    return sorted(
        filesystem.unstrip_protocol(path)
        for path in filesystem.glob(fs_pattern)
    )


def rebuild_cumulative_ground_truth(
    batch_files: list[str],
    *,
    output_path: str,
) -> pd.DataFrame:
    """
    Rebuild cumulative Ground Truth idempotently from all batches.

    Rebuilding is preferable here because it remains correct after
    retries and does not append the same batch more than once.
    """

    if not batch_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for batch_path in batch_files:
        with fsspec.open(
            batch_path,
            "rb",
        ) as file:
            content = file.read()

        batch_df = pd.read_csv(
            io.BytesIO(content),
            parse_dates=["Date"],
            dtype={"StateHoliday": str},
        )
        frames.append(batch_df)

    cumulative = pd.concat(
        frames,
        ignore_index=True,
    )

    cumulative["Date"] = pd.to_datetime(
        cumulative["Date"],
        errors="coerce",
    )
    cumulative["Store"] = pd.to_numeric(
        cumulative["Store"],
        errors="coerce",
    ).astype("Int64")

    cumulative = (
        cumulative.dropna(
            subset=["Store", "Date"]
        )
        .sort_values(["Date", "Store"])
        .drop_duplicates(
            subset=["Store", "Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    with fsspec.open(
        output_path,
        "w",
    ) as file:
        cumulative.to_csv(
            file,
            index=False,
        )

    return cumulative


def refresh_monitoring_signals(
    *,
    rolling_window: str = "7D",
    minimum_performance_samples: int = 500,
) -> MonitoringRefreshResult:
    """
    Refresh delayed-label performance and feature-drift evidence.

    Missing predictions or unmatched Ground Truth are normal operational
    states and do not crash the decision flow.
    """

    raw_path = get_path("raw_data")
    predictions_path = get_path(
        "predictions"
    )
    monitoring_path = get_path(
        "monitoring"
    )

    batch_pattern = join_uri(
        raw_path,
        "new_batches",
        "ground_truth_*.csv",
    )
    batch_files = _list_files(
        batch_pattern
    )

    cumulative_path = join_uri(
        monitoring_path,
        "cumulative_ground_truth.csv",
    )
    inference_log_path = join_uri(
        predictions_path,
        "inference_log.parquet",
    )
    performance_path = join_uri(
        monitoring_path,
        "performance_rolling.parquet",
    )

    cumulative = (
        rebuild_cumulative_ground_truth(
            batch_files,
            output_path=cumulative_path,
        )
    )

    performance_updated = False
    performance_rows = 0
    performance_reason = (
        "No Ground-Truth batches available."
    )

    if (
        not cumulative.empty
        and file_exists(inference_log_path)
    ):
        try:
            joined = (
                prepare_joined_evaluation_frame(
                    predictions_path=(
                        inference_log_path
                    ),
                    ground_truth_path=(
                        cumulative_path
                    ),
                    join_key=("Store", "Date"),
                    y_true_col="Sales",
                    y_pred_col="prediction",
                    time_col="Date",
                )
            )

            metrics = compute_rolling_metrics(
                df=joined,
                time_col="Date",
                window=rolling_window,
                y_true_col="Sales",
                y_pred_col="prediction",
                min_samples=(
                    minimum_performance_samples
                ),
            )

            if metrics.empty:
                performance_reason = (
                    "Not enough matched samples "
                    "for a performance window."
                )
            else:
                save_metrics(
                    metrics,
                    performance_path,
                )
                performance_updated = True
                performance_rows = len(metrics)
                performance_reason = (
                    "Performance history refreshed."
                )

        except ValueError as error:
            performance_reason = str(error)

    elif (
        not cumulative.empty
        and not file_exists(
            inference_log_path
        )
    ):
        performance_reason = (
            "No inference log available."
        )

    drift_result = run_feature_drift_check()

    return MonitoringRefreshResult(
        ground_truth_rows=len(cumulative),
        performance_updated=(
            performance_updated
        ),
        performance_rows=performance_rows,
        feature_drift_updated=(
            not drift_result.empty
        ),
        feature_drift_rows=len(
            drift_result
        ),
        performance_reason=(
            performance_reason
        ),
    )