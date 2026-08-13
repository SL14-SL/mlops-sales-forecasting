from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalEvaluation:
    triggered: bool
    window_end: str | None
    reason: str


def evaluate_persistent_feature_drift(
    history: pd.DataFrame,
    *,
    evaluated_at: pd.Timestamp,
    lookback_days: int,
    consecutive_windows: int,
) -> SignalEvaluation:
    """
    Evaluate drift only from recent complete drift checks.

    A drift signal is persistent when at least one feature is marked
    as drifted in every one of the latest N evaluation windows.
    """

    if history.empty:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason="No feature drift history available.",
        )
    
    required_columns = {
        "timestamp",
        "feature",
        "drift_detected",
    }

    missing_columns = (
        required_columns - set(history.columns)
    )
    if missing_columns:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason=(
                "Feature drift history is missing columns: "
                f"{sorted(missing_columns)}."
            ),
        )

    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["drift_detected"] = (
        frame["drift_detected"]
        .fillna(False)
        .astype(bool)
    )
    frame = frame.dropna(
        subset=["timestamp", "feature"]
    )

    evaluation_time = pd.Timestamp(evaluated_at)
    if evaluation_time.tzinfo is None:
        evaluation_time = evaluation_time.tz_localize(
            "UTC"
        )
    else:
        evaluation_time = evaluation_time.tz_convert(
            "UTC"
        )

    cutoff = evaluation_time - pd.Timedelta(
        days=lookback_days
    )

    frame = frame[
        (frame["timestamp"] >= cutoff)
        & (frame["timestamp"] <= evaluation_time)
    ]

    window_timestamps = (
        frame["timestamp"]
        .drop_duplicates()
        .sort_values()
        .tail(consecutive_windows)
        .tolist()
    )

    if len(window_timestamps) < consecutive_windows:
        return SignalEvaluation(
            triggered=False,
            window_end=(
                window_timestamps[-1].isoformat()
                if window_timestamps
                else None
            ),
            reason=(
                "Not enough recent feature drift windows: "
                f"{len(window_timestamps)}/"
                f"{consecutive_windows}."
            ),
        )

    recent = frame[
        frame["timestamp"].isin(
            window_timestamps
        )
    ]

    drift_by_feature = (
        recent.groupby("feature")[
            "drift_detected"
        ]
        .agg(["sum", "count"])
    )

    persistent_features = drift_by_feature[
        (
            drift_by_feature["count"]
            == consecutive_windows
        )
        & (
            drift_by_feature["sum"]
            == consecutive_windows
        )
    ].index.tolist()

    latest_window = pd.Timestamp(
        window_timestamps[-1]
    ).isoformat()

    if not persistent_features:
        return SignalEvaluation(
            triggered=False,
            window_end=latest_window,
            reason=(
                "No feature drift persisted across "
                f"{consecutive_windows} windows."
            ),
        )

    return SignalEvaluation(
        triggered=True,
        window_end=latest_window,
        reason=(
            "Persistent feature drift detected for: "
            f"{sorted(persistent_features)}."
        ),
    )


def evaluate_performance_degradation(
    history: pd.DataFrame,
    *,
    consecutive_windows: int,
    rmse_limit: float,
    mae_limit: float,
    absolute_bias_limit: float,
) -> SignalEvaluation:
    """
    Detect persistent degradation in the latest performance windows.

    A window is bad when RMSE and MAE both exceed their limits,
    or when absolute bias exceeds its limit.
    """

    if history.empty:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason="No performance history available.",
        )
    
    required_columns = {
        "window_end",
        "rmse",
        "mae",
        "bias",
    }

    missing_columns = (
        required_columns - set(history.columns)
    )
    if missing_columns:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason=(
                "Performance history is missing columns: "
                f"{sorted(missing_columns)}."
            ),
        )

    frame = history.copy()
    frame["window_end"] = pd.to_datetime(
        frame["window_end"],
        utc=True,
        errors="coerce",
    )

    for column in ["rmse", "mae", "bias"]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame.dropna(
            subset=[
                "window_end",
                "rmse",
                "mae",
                "bias",
            ]
        )
        .sort_values("window_end")
        .drop_duplicates(
            subset=["window_end"],
            keep="last",
        )
    )

    recent = frame.tail(consecutive_windows)

    window_end = (
        recent.iloc[-1]["window_end"].isoformat()
        if not recent.empty
        else None
    )

    if len(recent) < consecutive_windows:
        return SignalEvaluation(
            triggered=False,
            window_end=window_end,
            reason=(
                "Not enough performance windows: "
                f"{len(recent)}/"
                f"{consecutive_windows}."
            ),
        )

    bad_windows = (
        (
            (recent["rmse"] > rmse_limit)
            & (recent["mae"] > mae_limit)
        )
        | (
            recent["bias"].abs()
            > absolute_bias_limit
        )
    )

    if not bool(bad_windows.all()):
        return SignalEvaluation(
            triggered=False,
            window_end=window_end,
            reason=(
                "Performance degradation is not persistent "
                f"across {consecutive_windows} windows."
            ),
        )

    return SignalEvaluation(
        triggered=True,
        window_end=window_end,
        reason=(
            "Persistent performance degradation detected "
            f"across {consecutive_windows} windows."
        ),
    )