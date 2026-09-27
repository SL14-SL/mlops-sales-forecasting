from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..configs.paths import path_suffix
from ..storage.filesystem import (
    ensure_dir,
    file_exists,
)


def load_table(
    path: str | Path,
) -> pd.DataFrame:
    """Load a local or remote CSV or Parquet table."""
    path_string = str(path)

    if not file_exists(path_string):
        raise FileNotFoundError(f"Monitoring table not found: {path_string}")

    suffix = path_suffix(path_string)

    if suffix == ".parquet":
        return pd.read_parquet(path_string)

    if suffix == ".csv":
        return pd.read_csv(path_string)

    raise ValueError(f"Unsupported monitoring table format: {suffix!r}. Expected .csv or .parquet.")


def compute_regression_metrics(
    frame: pd.DataFrame,
    *,
    actual_column: str = "Sales",
    prediction_column: str = "prediction",
) -> dict[str, float | int]:
    """Compute regression metrics from matched observations."""
    required_columns = {
        actual_column,
        prediction_column,
    }
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise KeyError(f"Performance data is missing columns: {sorted(missing_columns)}.")

    clean_frame = frame.dropna(
        subset=[
            actual_column,
            prediction_column,
        ]
    ).copy()

    if clean_frame.empty:
        raise ValueError("No complete observations are available for performance evaluation.")

    actual = pd.to_numeric(
        clean_frame[actual_column],
        errors="coerce",
    )
    predicted = pd.to_numeric(
        clean_frame[prediction_column],
        errors="coerce",
    )

    valid_mask = actual.notna() & predicted.notna()
    actual = actual.loc[valid_mask].astype(float)
    predicted = predicted.loc[valid_mask].astype(float)

    if actual.empty:
        raise ValueError("Performance data contains no numeric actual and prediction pairs.")

    errors = actual - predicted

    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "n_samples": int(len(errors)),
    }


def _parse_rolling_window(
    window: str,
) -> pd.Timedelta:
    """Parse a positive rolling-window duration."""
    normalized_window = window.strip().lower()

    unit_mapping = {
        "d": "d",
        "h": "h",
        "m": "m",
    }

    if len(normalized_window) < 2:
        raise ValueError(
            f"Invalid rolling window: {window!r}."
        )

    unit = normalized_window[-1]
    value = normalized_window[:-1]

    if unit not in unit_mapping:
        raise ValueError(
            f"Invalid rolling-window unit: {unit!r}."
        )

    try:
        amount = float(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid rolling window: {window!r}."
        ) from error

    if amount <= 0:
        raise ValueError(
            "Rolling window must be positive."
        )

    return pd.to_timedelta(
        amount,
        unit=unit_mapping[unit],
    )


def compute_rolling_metrics(
    frame: pd.DataFrame,
    *,
    time_column: str = "Date",
    window: str = "7D",
    actual_column: str = "Sales",
    prediction_column: str = "prediction",
    minimum_samples: int = 1,
) -> pd.DataFrame:
    """Compute trailing performance metrics per timestamp."""
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive.")

    if time_column not in frame.columns:
        raise KeyError(f"Performance data has no {time_column!r} column.")

    window_delta = _parse_rolling_window(window)

    working_frame = frame.copy()
    working_frame[time_column] = pd.to_datetime(
        working_frame[time_column],
        errors="coerce",
    )
    working_frame = working_frame.dropna(
        subset=[
            time_column,
            actual_column,
            prediction_column,
        ]
    ).sort_values(time_column)

    if working_frame.empty:
        raise ValueError(
            "No valid timestamped observations are available for rolling performance evaluation."
        )

    rows: list[dict[str, object]] = []

    for window_end in working_frame[time_column].drop_duplicates().sort_values():
        window_start = window_end - window_delta
        window_frame = working_frame.loc[
            (working_frame[time_column] > window_start) & (working_frame[time_column] <= window_end)
        ]

        if len(window_frame) < minimum_samples:
            continue

        metrics = compute_regression_metrics(
            window_frame,
            actual_column=actual_column,
            prediction_column=prediction_column,
        )
        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                **metrics,
            }
        )

    columns = [
        "window_start",
        "window_end",
        "rmse",
        "mae",
        "bias",
        "n_samples",
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def prepare_evaluation_frame(
    *,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    join_columns: Sequence[str] = (
        "Store",
        "Date",
    ),
    actual_column: str = "Sales",
    prediction_column: str = "prediction",
    time_column: str | None = "Date",
) -> pd.DataFrame:
    """Join predictions with their corresponding ground truth."""
    normalized_join_columns = list(join_columns)

    if not normalized_join_columns:
        raise ValueError("At least one evaluation join column is required.")

    for column in normalized_join_columns:
        if column not in predictions.columns:
            raise KeyError(f"Predictions are missing join column {column!r}.")

        if column not in ground_truth.columns:
            raise KeyError(f"Ground truth is missing join column {column!r}.")

    if prediction_column not in predictions.columns:
        raise KeyError(f"Predictions are missing value column {prediction_column!r}.")

    if actual_column not in ground_truth.columns:
        raise KeyError(f"Ground truth is missing value column {actual_column!r}.")

    prediction_frame = predictions[
        [
            *normalized_join_columns,
            prediction_column,
        ]
    ].copy()
    ground_truth_frame = ground_truth[
        [
            *normalized_join_columns,
            actual_column,
        ]
    ].copy()

    if "Date" in normalized_join_columns:
        prediction_frame["Date"] = pd.to_datetime(
            prediction_frame["Date"],
            errors="coerce",
        )
        ground_truth_frame["Date"] = pd.to_datetime(
            ground_truth_frame["Date"],
            errors="coerce",
        )

    if "Store" in normalized_join_columns:
        prediction_frame["Store"] = pd.to_numeric(
            prediction_frame["Store"],
            errors="coerce",
        ).astype("Int64")
        ground_truth_frame["Store"] = pd.to_numeric(
            ground_truth_frame["Store"],
            errors="coerce",
        ).astype("Int64")

    prediction_frame = prediction_frame.dropna(subset=normalized_join_columns)
    ground_truth_frame = ground_truth_frame.dropna(subset=normalized_join_columns)

    joined_frame = prediction_frame.merge(
        ground_truth_frame,
        on=normalized_join_columns,
        how="inner",
        validate="one_to_one",
    )

    if joined_frame.empty:
        raise ValueError("Predictions and ground truth contain no matching observations.")

    if time_column is not None and time_column not in joined_frame.columns:
        raise KeyError(f"Joined performance data is missing time column {time_column!r}.")

    return joined_frame


def save_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Persist performance metrics as CSV or Parquet."""
    output_path_string = str(output_path)

    if not output_path_string.startswith("gs://"):
        ensure_dir(str(Path(output_path_string).parent))

    suffix = path_suffix(output_path_string)

    if suffix == ".parquet":
        metrics.to_parquet(
            output_path_string,
            index=False,
        )
        return

    if suffix == ".csv":
        metrics.to_csv(
            output_path_string,
            index=False,
        )
        return

    raise ValueError(
        f"Unsupported performance output format: {suffix!r}. Expected .csv or .parquet."
    )


def evaluate_predictions(
    *,
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    join_columns: Sequence[str] = (
        "Store",
        "Date",
    ),
    time_column: str | None = "Date",
    actual_column: str = "Sales",
    prediction_column: str = "prediction",
    rolling_window: str = "7D",
    minimum_samples: int = 1,
) -> pd.DataFrame:
    """Evaluate persisted predictions against ground truth."""
    predictions = load_table(predictions_path)
    ground_truth = load_table(ground_truth_path)

    joined_frame = prepare_evaluation_frame(
        predictions=predictions,
        ground_truth=ground_truth,
        join_columns=join_columns,
        actual_column=actual_column,
        prediction_column=prediction_column,
        time_column=time_column,
    )

    if time_column is None:
        metrics = pd.DataFrame(
            [
                compute_regression_metrics(
                    joined_frame,
                    actual_column=actual_column,
                    prediction_column=prediction_column,
                )
            ]
        )
    else:
        metrics = compute_rolling_metrics(
            joined_frame,
            time_column=time_column,
            window=rolling_window,
            actual_column=actual_column,
            prediction_column=prediction_column,
            minimum_samples=minimum_samples,
        )

    save_metrics(
        metrics,
        output_path,
    )
    return metrics
