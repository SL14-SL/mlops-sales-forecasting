import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIRECTORY = Path("results")


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    """Return the first available column from a list of candidates."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        "None of the expected columns were found: "
        f"{candidates}. Available columns: "
        f"{frame.columns.tolist()}"
    )


def load_lifecycle(path: Path) -> pd.DataFrame:
    """Load and normalize lifecycle monitoring results."""
    frame = pd.read_csv(path)

    required_columns = {
        "day",
        "rmse",
        "event",
    }

    missing_columns = required_columns.difference(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing lifecycle columns in {path}: "
            f"{sorted(missing_columns)}"
        )

    frame["day"] = pd.to_numeric(
        frame["day"],
        errors="raise",
    ).astype(int)

    frame["rmse"] = pd.to_numeric(
        frame["rmse"],
        errors="coerce",
    )

    return (
        frame
        .sort_values("day")
        .reset_index(drop=True)
    )


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and normalize prediction records."""
    frame = pd.read_parquet(path)

    date_column = find_column(
        frame,
        ["Date", "date"],
    )
    store_column = find_column(
        frame,
        ["Store", "store"],
    )
    prediction_column = find_column(
        frame,
        [
            "prediction",
            "Prediction",
            "y_pred",
        ],
    )

    frame = frame.rename(
        columns={
            date_column: "Date",
            store_column: "Store",
            prediction_column: "prediction",
        }
    )

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        errors="raise",
    )
    frame["Store"] = pd.to_numeric(
        frame["Store"],
        errors="raise",
    ).astype(int)
    frame["prediction"] = pd.to_numeric(
        frame["prediction"],
        errors="raise",
    )

    return (
        frame[
            [
                "Store",
                "Date",
                "prediction",
            ]
        ]
        .sort_values(["Date", "Store"])
        .drop_duplicates(
            subset=["Store", "Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load and normalize delayed ground-truth records."""
    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    date_column = find_column(
        frame,
        ["Date", "date"],
    )
    store_column = find_column(
        frame,
        ["Store", "store"],
    )
    sales_column = find_column(
        frame,
        ["Sales", "sales"],
    )
    promo_column = find_column(
        frame,
        ["Promo", "promo"],
    )
    open_column = find_column(
        frame,
        ["Open", "open"],
    )

    frame = frame.rename(
        columns={
            date_column: "Date",
            store_column: "Store",
            sales_column: "Sales",
            promo_column: "Promo",
            open_column: "Open",
        }
    )

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        errors="raise",
    )
    frame["Store"] = pd.to_numeric(
        frame["Store"],
        errors="raise",
    ).astype(int)

    for column in [
        "Sales",
        "Promo",
        "Open",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return (
        frame[
            [
                "Store",
                "Date",
                "Sales",
                "Promo",
                "Open",
            ]
        ]
        .sort_values(["Date", "Store"])
        .drop_duplicates(
            subset=["Store", "Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def merge_predictions_and_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Merge predictions with matching open-store ground truth."""
    merged = predictions.merge(
        ground_truth,
        on=["Store", "Date"],
        how="inner",
        validate="one_to_one",
    )

    return merged.loc[
        merged["Open"].eq(1)
        & merged["Sales"].notna()
        & merged["prediction"].notna()
    ].copy()


def align_evaluation_records(
    without_frame: pd.DataFrame,
    with_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep identical store-date observations in both variants."""
    common_keys = (
        without_frame[
            ["Store", "Date"]
        ]
        .merge(
            with_frame[
                ["Store", "Date"]
            ],
            on=["Store", "Date"],
            how="inner",
        )
        .drop_duplicates()
    )

    aligned_without = without_frame.merge(
        common_keys,
        on=["Store", "Date"],
        how="inner",
        validate="one_to_one",
    )

    aligned_with = with_frame.merge(
        common_keys,
        on=["Store", "Date"],
        how="inner",
        validate="one_to_one",
    )

    return aligned_without, aligned_with


def calculate_metrics(
    frame: pd.DataFrame,
) -> dict:
    """Calculate regression metrics on the original sales scale."""
    if frame.empty:
        return {
            "rows": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "wmape_percent": np.nan,
            "bias": np.nan,
        }

    errors = (
        frame["prediction"]
        - frame["Sales"]
    )

    absolute_sales_sum = (
        frame["Sales"].abs().sum()
    )

    wmape_percent = np.nan

    if absolute_sales_sum > 0:
        wmape_percent = (
            errors.abs().sum()
            / absolute_sales_sum
            * 100.0
        )

    return {
        "rows": len(frame),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(errors)
                )
            )
        ),
        "mae": float(
            np.mean(
                np.abs(errors)
            )
        ),
        "wmape_percent": float(
            wmape_percent
        ),
        "bias": float(
            np.mean(errors)
        ),
    }


def build_segment_metrics(
    model_variant: str,
    frame: pd.DataFrame,
) -> list[dict]:
    """Calculate metrics for overall, promo and non-promo segments."""
    segments = {
        "All open stores": frame,
        "Promo stores": frame.loc[
            frame["Promo"].eq(1)
        ],
        "Non-promo stores": frame.loc[
            frame["Promo"].ne(1)
        ],
    }

    records = []

    for segment_name, segment_frame in segments.items():
        records.append(
            {
                "model_variant": model_variant,
                "segment": segment_name,
                **calculate_metrics(
                    segment_frame
                ),
            }
        )

    return records


def calculate_rmse_change(
    baseline: float,
    candidate: float,
) -> float:
    """Calculate relative RMSE change where negative is better."""
    if baseline == 0:
        return np.nan

    return (
        (candidate - baseline)
        / baseline
        * 100.0
    )


def find_promotion_day(
    lifecycle: pd.DataFrame,
) -> int:
    """Return the first successful retraining day."""
    retrain_rows = lifecycle.loc[
        lifecycle["event"].eq("retrain")
    ]

    if retrain_rows.empty:
        raise ValueError(
            "No retrain event was found in the "
            "with-retraining lifecycle file."
        )

    if "champion_promoted" in retrain_rows.columns:
        promoted = retrain_rows.loc[
            retrain_rows[
                "champion_promoted"
            ].fillna(False).astype(bool)
        ]

        if not promoted.empty:
            retrain_rows = promoted

    return int(
        retrain_rows.iloc[0]["day"]
    )


def get_segment_metric(
    metrics: pd.DataFrame,
    model_variant: str,
    segment: str,
    metric: str,
) -> float:
    """Return one metric value for a model and segment."""
    selection = metrics.loc[
        (
            metrics["model_variant"]
            == model_variant
        )
        & (
            metrics["segment"]
            == segment
        ),
        metric,
    ]

    if selection.empty:
        raise ValueError(
            "Metric not found for "
            f"model={model_variant}, "
            f"segment={segment}, "
            f"metric={metric}."
        )

    return float(
        selection.iloc[0]
    )


def create_comparison_plot(
    *,
    without_lifecycle_path: Path,
    without_predictions_path: Path,
    without_ground_truth_path: Path,
    with_lifecycle_path: Path,
    with_predictions_path: Path,
    with_ground_truth_path: Path,
    metrics_output_path: Path,
    plot_output_path: Path,
    drift_start_day: int,
    drift_duration_days: int,
    rolling_window_days: int,
) -> None:
    """Create a lifecycle and segment comparison for mild weighting."""
    without_lifecycle = load_lifecycle(
        without_lifecycle_path
    )
    with_lifecycle = load_lifecycle(
        with_lifecycle_path
    )

    promotion_day = find_promotion_day(
        with_lifecycle
    )

    without_predictions = load_predictions(
        without_predictions_path
    )
    with_predictions = load_predictions(
        with_predictions_path
    )

    without_ground_truth = load_ground_truth(
        without_ground_truth_path
    )
    with_ground_truth = load_ground_truth(
        with_ground_truth_path
    )

    with_dates = sorted(
        with_ground_truth["Date"].unique()
    )

    if len(with_dates) < promotion_day:
        raise ValueError(
            "Promotion day exceeds the available "
            "ground-truth date range."
        )

    promotion_date = pd.Timestamp(
        with_dates[promotion_day - 1]
    )
    evaluation_start_date = (
        promotion_date
        + pd.Timedelta(days=1)
    )

    without_records = merge_predictions_and_truth(
        without_predictions,
        without_ground_truth,
    )
    with_records = merge_predictions_and_truth(
        with_predictions,
        with_ground_truth,
    )

    without_post = without_records.loc[
        without_records["Date"]
        >= evaluation_start_date
    ].copy()

    with_post = with_records.loc[
        with_records["Date"]
        >= evaluation_start_date
    ].copy()

    without_post, with_post = (
        align_evaluation_records(
            without_post,
            with_post,
        )
    )

    metric_records = []

    metric_records.extend(
        build_segment_metrics(
            "Without retraining",
            without_post,
        )
    )
    metric_records.extend(
        build_segment_metrics(
            "With mild-weight final refit",
            with_post,
        )
    )

    metrics = pd.DataFrame(
        metric_records
    )

    metrics_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        metrics_output_path,
        index=False,
    )

    lifecycle = (
        without_lifecycle[
            ["day", "rmse"]
        ]
        .merge(
            with_lifecycle[
                ["day", "rmse"]
            ],
            on="day",
            how="inner",
            suffixes=(
                "_without",
                "_with",
            ),
        )
        .sort_values("day")
        .reset_index(drop=True)
    )

    first_full_post_window_day = (
        promotion_day
        + rolling_window_days
    )

    post_rolling = lifecycle.loc[
        lifecycle["day"]
        >= first_full_post_window_day
    ]

    mean_rolling_without = float(
        post_rolling[
            "rmse_without"
        ].mean()
    )
    mean_rolling_with = float(
        post_rolling[
            "rmse_with"
        ].mean()
    )

    rolling_rmse_change = (
        calculate_rmse_change(
            mean_rolling_without,
            mean_rolling_with,
        )
    )

    segment_order = [
        "All open stores",
        "Promo stores",
        "Non-promo stores",
    ]

    without_rmse = []
    with_rmse = []
    rmse_changes = []

    for segment in segment_order:
        without_value = get_segment_metric(
            metrics=metrics,
            model_variant="Without retraining",
            segment=segment,
            metric="rmse",
        )

        with_value = get_segment_metric(
            metrics=metrics,
            model_variant=(
                "With mild-weight final refit"
            ),
            segment=segment,
            metric="rmse",
        )

        without_rmse.append(
            without_value
        )
        with_rmse.append(
            with_value
        )
        rmse_changes.append(
            calculate_rmse_change(
                without_value,
                with_value,
            )
        )

    plt.style.use(
        "seaborn-v0_8-whitegrid"
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(16, 12),
        gridspec_kw={
            "height_ratios": [
                2.2,
                1.0,
            ],
        },
    )

    line_axis = axes[0]
    bar_axis = axes[1]

    line_axis.plot(
        lifecycle["day"],
        lifecycle["rmse_without"],
        color="#ef553b",
        linestyle="--",
        linewidth=2.5,
        label="Without retraining",
    )

    line_axis.plot(
        lifecycle["day"],
        lifecycle["rmse_with"],
        color="#636efa",
        linewidth=3.0,
        label=(
            "With mild-weight final-refit retraining"
        ),
    )

    full_drift_day = (
        drift_start_day
        + drift_duration_days
        - 1
    )

    line_axis.axvspan(
        drift_start_day,
        full_drift_day,
        color="#f2c14e",
        alpha=0.2,
        label="Promo-effect decay ramp-up",
    )

    line_axis.axvline(
        full_drift_day,
        color="#ef553b",
        linestyle=":",
        linewidth=2,
        label="Full promo-effect decay",
    )

    line_axis.axvline(
        promotion_day,
        color="#00a67d",
        linestyle="--",
        linewidth=2,
        label="Retraining and final refit",
    )

    line_axis.axvline(
        first_full_post_window_day,
        color="#7f8c8d",
        linestyle=":",
        linewidth=2,
        label=(
            "First full post-retraining RMSE window"
        ),
    )

    line_axis.set_title(
        "Rolling Forecast Error During Controlled Promo-Effect Drift",
        fontsize=16,
    )
    line_axis.set_xlabel(
        "Simulation day"
    )
    line_axis.set_ylabel(
        f"{rolling_window_days}-day rolling RMSE"
    )
    line_axis.grid(
        alpha=0.25
    )
    line_axis.legend(
        loc="upper right"
    )

    rolling_summary = (
        f"Mean rolling RMSE from day "
        f"{first_full_post_window_day}\n"
        f"Without retraining: "
        f"{mean_rolling_without:.1f}\n"
        f"With mild weighting: "
        f"{mean_rolling_with:.1f}\n"
        f"Relative RMSE change: "
        f"{rolling_rmse_change:+.1f}%"
    )

    line_axis.text(
        0.02,
        0.96,
        rolling_summary,
        transform=line_axis.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "#495057",
            "alpha": 0.92,
        },
    )

    positions = np.arange(
        len(segment_order)
    )
    bar_width = 0.36

    without_bars = bar_axis.bar(
        positions - bar_width / 2,
        without_rmse,
        width=bar_width,
        color="#ef553b",
        alpha=0.9,
        label="Without retraining",
    )

    with_bars = bar_axis.bar(
        positions + bar_width / 2,
        with_rmse,
        width=bar_width,
        color="#636efa",
        alpha=0.9,
        label=(
            "With mild-weight final refit"
        ),
    )

    bar_axis.bar_label(
        without_bars,
        fmt="%.0f",
        padding=3,
    )
    bar_axis.bar_label(
        with_bars,
        fmt="%.0f",
        padding=3,
    )

    maximum_bar_value = max(
        without_rmse + with_rmse
    )

    for index, rmse_change in enumerate(
        rmse_changes
    ):
        annotation_color = (
            "#087f5b"
            if rmse_change < 0
            else "#c92a2a"
        )

        bar_axis.text(
            index,
            max(
                without_rmse[index],
                with_rmse[index],
            )
            + maximum_bar_value * 0.08,
            f"RMSE {rmse_change:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=annotation_color,
        )

    bar_axis.axhline(
        y=0,
        color="#495057",
        linewidth=0.8,
    )

    bar_axis.set_title(
        "Post-Promotion RMSE by Forecast Segment",
        fontsize=15,
    )
    bar_axis.set_ylabel(
        "RMSE"
    )
    bar_axis.set_xticks(
        positions,
        segment_order,
    )
    bar_axis.set_ylim(
        0,
        maximum_bar_value * 1.25,
    )
    bar_axis.grid(
        axis="y",
        alpha=0.25,
    )
    bar_axis.legend(
        loc="upper right"
    )

    figure.suptitle(
        "Promo-Effect Decay: Performance With and Without Retraining",
        fontsize=19,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.01,
        (
            "Negative relative RMSE change indicates lower forecast error; "
            "positive change indicates higher forecast error."
        ),
        ha="center",
        fontsize=10,
        color="#495057",
    )

    figure.tight_layout(
        rect=[
            0,
            0.03,
            1,
            0.97,
        ]
    )

    plot_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        plot_output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print()
    print(
        f"Promotion day: {promotion_day}"
    )
    print(
        "Promotion date: "
        f"{promotion_date.date()}"
    )
    print(
        "Post-promotion evaluation starts: "
        f"{evaluation_start_date.date()}"
    )
    print(
        "Post-promotion evaluation days: "
        f"{with_post['Date'].nunique()}"
    )
    print()
    print(
        metrics.to_string(
            index=False,
            formatters={
                "rmse": "{:.2f}".format,
                "mae": "{:.2f}".format,
                "wmape_percent": "{:.2f}".format,
                "bias": "{:.2f}".format,
            },
        )
    )
    print()
    print(
        "Relative RMSE change by segment:"
    )

    for segment, rmse_change in zip(
        segment_order,
        rmse_changes,
        strict=True,
    ):
        print(
            f"  {segment}: "
            f"{rmse_change:+.2f}%"
        )

    print()
    print(
        f"Rolling RMSE change: "
        f"{rolling_rmse_change:+.2f}%"
    )
    print(
        f"Metrics saved to: "
        f"{metrics_output_path}"
    )
    print(
        f"Plot saved to: "
        f"{plot_output_path}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare a mild recency-weighted final-refit model "
            "with a no-retraining baseline."
        )
    )

    parser.add_argument(
        "--without-lifecycle",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_weighted_without_retraining.csv"
        ),
    )
    parser.add_argument(
        "--without-predictions",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_weighted_without_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--without-ground-truth",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_weighted_without_ground_truth.csv"
        ),
    )
    parser.add_argument(
        "--with-lifecycle",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_mild_weights_with_retraining.csv"
        ),
    )
    parser.add_argument(
        "--with-predictions",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_mild_weights_with_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--with-ground-truth",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_mild_weights_with_ground_truth.csv"
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_mild_weights_segment_metrics.csv"
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "promo_mild_weights_comparison.png"
        ),
    )
    parser.add_argument(
        "--drift-start-day",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--drift-duration-days",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=7,
    )

    return parser.parse_args()


def main() -> None:
    """Run the retraining comparison."""
    args = parse_args()

    create_comparison_plot(
        without_lifecycle_path=(
            args.without_lifecycle
        ),
        without_predictions_path=(
            args.without_predictions
        ),
        without_ground_truth_path=(
            args.without_ground_truth
        ),
        with_lifecycle_path=(
            args.with_lifecycle
        ),
        with_predictions_path=(
            args.with_predictions
        ),
        with_ground_truth_path=(
            args.with_ground_truth
        ),
        metrics_output_path=(
            args.metrics_output
        ),
        plot_output_path=(
            args.plot_output
        ),
        drift_start_day=(
            args.drift_start_day
        ),
        drift_duration_days=(
            args.drift_duration_days
        ),
        rolling_window_days=(
            args.rolling_window_days
        ),
    )


if __name__ == "__main__":
    main()