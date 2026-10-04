from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_sales_forecasting.monitoring.monitoring_refresh import (
    load_inference_history,
)
from mlops_sales_forecasting.monitoring.performance import (
    compute_regression_metrics,
)


def build_final_evaluation_frame(
    *,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """Build the final open-store evaluation window."""
    if window_days < 1:
        raise ValueError("Evaluation window must be positive.")

    required_prediction_columns = {
        "Store",
        "Date",
        "prediction",
        "release_id",
        "timestamp",
    }
    missing_predictions = required_prediction_columns - set(predictions.columns)

    if missing_predictions:
        raise KeyError(
            f"Simulation predictions are missing columns: {sorted(missing_predictions)}."
        )

    required_ground_truth_columns = {
        "Store",
        "Date",
        "Sales",
        "Open",
        "Promo",
    }
    missing_ground_truth = required_ground_truth_columns - set(ground_truth.columns)

    if missing_ground_truth:
        raise KeyError(
            f"Simulation Ground Truth is missing columns: {sorted(missing_ground_truth)}."
        )

    prediction_frame = predictions.copy()
    prediction_frame["Store"] = pd.to_numeric(
        prediction_frame["Store"],
        errors="coerce",
    ).astype("Int64")
    prediction_frame["Date"] = pd.to_datetime(
        prediction_frame["Date"],
        errors="coerce",
    )
    prediction_frame["timestamp"] = pd.to_datetime(
        prediction_frame["timestamp"],
        errors="coerce",
        utc=True,
    )

    prediction_frame = (
        prediction_frame.dropna(
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
                "Store",
                "Date",
            ],
            keep="last",
        )
    )

    ground_truth_frame = ground_truth.copy()
    ground_truth_frame["Store"] = pd.to_numeric(
        ground_truth_frame["Store"],
        errors="coerce",
    ).astype("Int64")
    ground_truth_frame["Date"] = pd.to_datetime(
        ground_truth_frame["Date"],
        errors="coerce",
    )

    evaluation = prediction_frame[
        [
            "Store",
            "Date",
            "prediction",
            "release_id",
        ]
    ].merge(
        ground_truth_frame[
            [
                "Store",
                "Date",
                "Sales",
                "Open",
                "Promo",
            ]
        ],
        on=[
            "Store",
            "Date",
        ],
        how="inner",
        validate="one_to_one",
    )

    if evaluation.empty:
        raise ValueError(
            "Simulation predictions and Ground Truth contain no matching observations."
        )

    window_end = pd.Timestamp(evaluation["Date"].max())
    window_start = pd.Timestamp(
        window_end.to_pydatetime()
        - timedelta(
            days=window_days,
        )
    )

    evaluation = (
        evaluation.loc[
            evaluation["Date"].gt(window_start)
            & evaluation["Date"].le(window_end)
            & pd.to_numeric(
                evaluation["Open"],
                errors="coerce",
            ).eq(1)
        ]
        .sort_values(
            [
                "Date",
                "Store",
            ]
        )
        .reset_index(drop=True)
    )

    if evaluation.empty:
        raise ValueError("The final evaluation window contains no open-store observations.")

    return evaluation


def build_segment_metrics(
    evaluation: pd.DataFrame,
    *,
    model_variant: str,
) -> pd.DataFrame:
    """Calculate final metrics for operational segments."""
    if not model_variant.strip():
        raise ValueError("Model variant must not be empty.")

    promo = pd.to_numeric(
        evaluation["Promo"],
        errors="coerce",
    )

    segments = (
        (
            "All open stores",
            pd.Series(
                True,
                index=evaluation.index,
            ),
        ),
        (
            "Promo stores",
            promo.eq(1),
        ),
        (
            "Non-promo stores",
            promo.eq(0),
        ),
    )

    rows: list[dict[str, object]] = []

    for segment_name, mask in segments:
        segment = evaluation.loc[mask].copy()

        if segment.empty:
            raise ValueError(f"Evaluation segment contains no rows: {segment_name}.")

        metrics = compute_regression_metrics(
            segment,
            actual_column="Sales",
            prediction_column="prediction",
        )

        actual = pd.to_numeric(
            segment["Sales"],
            errors="coerce",
        ).astype(float)
        predicted = pd.to_numeric(
            segment["prediction"],
            errors="coerce",
        ).astype(float)

        denominator = float(np.abs(actual).sum())

        if denominator == 0:
            raise ValueError(
                f"WMAPE is undefined because the actual sum is zero for {segment_name}."
            )

        wmape = float(np.abs(actual - predicted).sum() / denominator * 100)

        rows.append(
            {
                "model_variant": model_variant,
                "segment": segment_name,
                "rows": int(metrics["n_samples"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "wmape_percent": wmape,
                "bias": float(metrics["bias"]),
            }
        )

    return pd.DataFrame(rows)


def export_runtime_evaluation(
    *,
    runtime_root: str | Path,
    output_path: str | Path,
    window_days: int = 7,
) -> pd.DataFrame:
    """Export the final evaluation window from a runtime."""
    runtime = Path(runtime_root)
    prediction_files = [str(path) for path in sorted((runtime / "predictions").rglob("*.parquet"))]

    if not prediction_files:
        raise FileNotFoundError(
            f"No simulation prediction history found below: {runtime / 'predictions'}"
        )

    ground_truth_path = runtime / "monitoring" / "cumulative_ground_truth.csv"

    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Simulation cumulative Ground Truth is missing: {ground_truth_path}"
        )

    predictions = load_inference_history(prediction_files)
    ground_truth = pd.read_csv(
        ground_truth_path,
        parse_dates=[
            "Date",
        ],
        dtype={
            "StateHoliday": str,
        },
    )

    evaluation = build_final_evaluation_frame(
        predictions=predictions,
        ground_truth=ground_truth,
        window_days=window_days,
    )

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    evaluation.to_parquet(
        destination,
        index=False,
    )

    return evaluation
