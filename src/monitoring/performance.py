from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Union, List

import numpy as np
import pandas as pd


# -------------------------------------------------
# I/O Helpers
# -------------------------------------------------

def load_table(path: str | Path) -> pd.DataFrame:
    """
    Load a table from parquet or CSV based on file suffix.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {suffix}. Use .parquet or .csv")


# -------------------------------------------------
# Metrics Computation
# -------------------------------------------------

def compute_regression_metrics(
    df: pd.DataFrame,
    y_true_col: str = "Sales",
    y_pred_col: str = "prediction",
) -> Dict[str, float]:
    """
    Compute standard regression metrics on a dataframe.

    Expected columns by default:
    - y_true_col: Sales
    - y_pred_col: prediction
    """
    if y_true_col not in df.columns:
        raise KeyError(f"Missing ground truth column: {y_true_col}")
    if y_pred_col not in df.columns:
        raise KeyError(f"Missing prediction column: {y_pred_col}")

    clean_df = df.dropna(subset=[y_true_col, y_pred_col]).copy()

    if clean_df.empty:
        raise ValueError("No rows available after dropping missing values.")

    y_true = clean_df[y_true_col].astype(float)
    y_pred = clean_df[y_pred_col].astype(float)

    errors = y_true - y_pred

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias),
        "n_samples": int(len(clean_df)),
    }


# -------------------------------------------------
# Rolling Window Metrics
# -------------------------------------------------

def compute_rolling_metrics(
    df: pd.DataFrame,
    time_col: str = "Date",
    window: str = "7D",
    y_true_col: str = "Sales",
    y_pred_col: str = "prediction",
    min_samples: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling regression metrics over a time-based window.

    For each unique timestamp in time_col, metrics are computed over:
        (current_time - window, current_time]

    Example:
        window="7D" -> trailing 7-day window

    Returns one row per window endpoint.
    """
    if time_col not in df.columns:
        raise KeyError(f"Missing time column: {time_col}")

    working_df = df.copy()
    working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
    working_df = working_df.dropna(subset=[time_col, y_true_col, y_pred_col])
    working_df = working_df.sort_values(time_col)

    if working_df.empty:
        raise ValueError("No valid rows available for rolling metrics computation.")

    window_delta = pd.Timedelta(window)
    metrics_list = []

    unique_times = working_df[time_col].sort_values().drop_duplicates()

    for end_time in unique_times:
        start_time = end_time - window_delta
        window_df = working_df[
            (working_df[time_col] > start_time) &
            (working_df[time_col] <= end_time)
        ].copy()

        if len(window_df) < min_samples:
            continue

        metrics = compute_regression_metrics(
            window_df,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
        )
        metrics["window_start"] = start_time
        metrics["window_end"] = end_time
        metrics_list.append(metrics)

    if not metrics_list:
        return pd.DataFrame(
            columns=["window_start", "window_end", "rmse", "mae", "bias", "n_samples"]
        )

    return pd.DataFrame(metrics_list)


# -------------------------------------------------
# Save Metrics
# -------------------------------------------------

def save_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save metrics as parquet or CSV depending on output suffix.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".parquet":
        metrics.to_parquet(output_path, index=False)
        return
    if suffix == ".csv":
        metrics.to_csv(output_path, index=False)
        return

    raise ValueError(f"Unsupported output format: {suffix}. Use .parquet or .csv")


# -------------------------------------------------
# Data Preparation Helpers
# -------------------------------------------------

def prepare_joined_evaluation_frame(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    join_key: Union[str, List[str]] = ("Store", "Date"),
    y_true_col: str = "Sales",
    y_pred_col: str = "prediction",
    time_col: Optional[str] = "Date",
) -> pd.DataFrame:
    """
    Load predictions and ground truth, harmonize datatypes, and join them.

    By default, joins on:
    - Store
    - Date
    """
    preds = load_table(predictions_path)
    gt = load_table(ground_truth_path)

    join_keys = [join_key] if isinstance(join_key, str) else list(join_key)

    for key in join_keys:
        if key not in preds.columns:
            raise KeyError(f"Missing join key '{key}' in predictions.")
        if key not in gt.columns:
            raise KeyError(f"Missing join key '{key}' in ground truth.")

    # Harmonize common keys used in your project
    if "Date" in join_keys:
        preds["Date"] = pd.to_datetime(preds["Date"], errors="coerce")
        gt["Date"] = pd.to_datetime(gt["Date"], errors="coerce")

    if "Store" in join_keys:
        preds["Store"] = pd.to_numeric(preds["Store"], errors="coerce").astype("Int64")
        gt["Store"] = pd.to_numeric(gt["Store"], errors="coerce").astype("Int64")

    preds = preds.dropna(subset=join_keys)
    gt = gt.dropna(subset=join_keys)

    df = preds.merge(gt, on=join_keys, how="inner")

    if df.empty:
        raise ValueError(
            "No matching rows found between predictions and ground truth after join."
        )

    if y_true_col not in df.columns:
        raise KeyError(f"Ground truth column '{y_true_col}' not found after merge.")
    if y_pred_col not in df.columns:
        raise KeyError(f"Prediction column '{y_pred_col}' not found after merge.")

    if time_col is not None and time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found after merge.")

    return df


# -------------------------------------------------
# Full Evaluation Pipeline
# -------------------------------------------------

def evaluate_predictions(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    output_metrics_path: str | Path,
    join_key: Union[str, List[str]] = ("Store", "Date"),
    time_col: Optional[str] = None,
    y_true_col: str = "Sales",
    y_pred_col: str = "prediction",
    rolling_window: str = "7D",
    min_samples: int = 1,
) -> pd.DataFrame:
    """
    Load predictions + ground truth, compute metrics, and save results.

    If time_col is provided:
        rolling metrics are computed

    Otherwise:
        one global metrics row is computed
    """
    df = prepare_joined_evaluation_frame(
        predictions_path=predictions_path,
        ground_truth_path=ground_truth_path,
        join_key=join_key,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        time_col=time_col,
    )

    if time_col:
        metrics = compute_rolling_metrics(
            df=df,
            time_col=time_col,
            window=rolling_window,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            min_samples=min_samples,
        )
    else:
        metrics_dict = compute_regression_metrics(
            df=df,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
        )
        metrics = pd.DataFrame([metrics_dict])

    save_metrics(metrics, output_metrics_path)
    return metrics


# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model performance evaluation.")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions parquet/csv file",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth parquet/csv file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save computed metrics (.parquet or .csv)",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Optional time column for rolling metrics, e.g. Date",
    )
    parser.add_argument(
        "--rolling-window",
        type=str,
        default="7D",
        help="Rolling time window, e.g. 7D, 14D, 30D",
    )
    parser.add_argument(
        "--y-true-col",
        type=str,
        default="Sales",
        help="Ground truth column name",
    )
    parser.add_argument(
        "--y-pred-col",
        type=str,
        default="prediction",
        help="Prediction column name",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples required per rolling window",
    )

    args = parser.parse_args()

    print("🚀 Running performance evaluation...")

    metrics = evaluate_predictions(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        output_metrics_path=args.output,
        time_col=args.time_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        rolling_window=args.rolling_window,
        min_samples=args.min_samples,
    )

    print("\n📊 Computed Metrics:")
    print(metrics)

    print("\n✅ Done.")