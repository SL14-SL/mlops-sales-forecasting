from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from mlops_sales_forecasting.monitoring.theme import (
    CHART_VALUE_FONT_SIZE,
    GREEN,
    LIFECYCLE_COLORS,
    MINT,
    MUTED_GREEN,
    REJECTED,
    TEXT,
    WHITE,
    apply_chart_theme,
)
from mlops_sales_forecasting.simulation.reporting import (
    summarize_simulation_comparison,
)

_REQUIRED_LIFECYCLE_COLUMNS = {
    "day",
    "rmse",
    "mae",
    "bias",
}

_REQUIRED_SEGMENT_COLUMNS = {
    "model_variant",
    "segment",
    "rows",
    "rmse",
    "mae",
    "wmape_percent",
    "bias",
}


def load_simulation_frame(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate one lifecycle-simulation result."""
    result_path = Path(path)

    if not result_path.is_file():
        raise FileNotFoundError(f"Simulation result not found: {result_path}")

    frame = pd.read_csv(result_path)

    missing_columns = _REQUIRED_LIFECYCLE_COLUMNS - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Simulation result is missing required columns: {sorted(missing_columns)}."
        )

    if frame.empty:
        raise ValueError("Simulation result must not be empty.")

    normalized = frame.copy()
    normalized["day"] = pd.to_numeric(
        normalized["day"],
        errors="coerce",
    )

    for column in (
        "rmse",
        "mae",
        "bias",
    ):
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized = normalized.dropna(
        subset=[
            "day",
            "rmse",
            "mae",
            "bias",
        ]
    ).sort_values("day")

    if normalized.empty:
        raise ValueError("Simulation result contains no valid metric rows.")

    normalized["day"] = normalized["day"].astype(int)

    return normalized.reset_index(drop=True)


def load_segment_metrics(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate simulation segment metrics."""
    result_path = Path(path)

    if not result_path.is_file():
        raise FileNotFoundError(f"Segment metrics not found: {result_path}")

    frame = pd.read_csv(result_path)

    missing_columns = _REQUIRED_SEGMENT_COLUMNS - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Segment metrics are missing required columns: {sorted(missing_columns)}."
        )

    if frame.empty:
        raise ValueError("Segment metrics must not be empty.")

    return frame


def load_reference_simulation(
    base_path: str | Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load all checked-in lifecycle reference results."""
    base = Path(base_path)

    return (
        load_simulation_frame(base / "without_retraining.csv"),
        load_simulation_frame(base / "with_retraining.csv"),
        load_segment_metrics(base / "segment_metrics.csv"),
    )


def build_simulation_summary(
    without_retraining: pd.DataFrame,
    with_retraining: pd.DataFrame,
) -> dict[str, float | int]:
    """Build headline metrics for the comparison."""
    return summarize_simulation_comparison(
        without_retraining,
        with_retraining,
    )


def build_drift_summary(
    lifecycle: pd.DataFrame,
) -> dict[str, float | int]:
    """Extract the controlled-drift scenario metadata."""
    required_columns = {
        "drift_start_day",
        "drift_duration_days",
        "maximum_base_uplift",
        "maximum_promo_uplift",
    }
    missing_columns = required_columns - set(lifecycle.columns)

    if missing_columns:
        raise KeyError(f"Lifecycle result is missing drift metadata: {sorted(missing_columns)}.")

    values = lifecycle.iloc[0]
    start_day = int(values["drift_start_day"])
    duration_days = int(values["drift_duration_days"])

    return {
        "drift_start_day": start_day,
        "drift_duration_days": duration_days,
        "full_drift_day": (start_day + duration_days),
        "maximum_base_uplift": float(values["maximum_base_uplift"]),
        "maximum_promo_uplift": float(values["maximum_promo_uplift"]),
    }


def _add_drift_regions(
    figure: go.Figure,
    lifecycle: pd.DataFrame,
) -> None:
    """Mark the ramp and full-drift periods."""
    summary = build_drift_summary(lifecycle)
    start_day = int(summary["drift_start_day"])
    full_drift_day = int(summary["full_drift_day"])
    maximum_day = float(
        pd.to_numeric(
            lifecycle["day"],
            errors="coerce",
        ).max()
    )

    if maximum_day >= start_day:
        figure.add_vrect(
            x0=start_day,
            x1=min(
                full_drift_day,
                maximum_day + 0.5,
            ),
            fillcolor=MINT,
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text="Drift ramp",
            annotation_position="top left",
        )

    if maximum_day >= full_drift_day:
        figure.add_vrect(
            x0=full_drift_day,
            x1=maximum_day + 0.5,
            fillcolor=REJECTED,
            opacity=0.06,
            layer="below",
            line_width=0,
            annotation_text=("Full promotional drift"),
            annotation_position="top left",
        )


def build_lifecycle_metric_chart(
    without_retraining: pd.DataFrame,
    with_retraining: pd.DataFrame,
    *,
    metric: str,
) -> go.Figure:
    """Compare one lifecycle metric across both variants."""
    supported_metrics = {
        "rmse": "RMSE",
        "mae": "MAE",
        "bias": "Bias",
    }

    if metric not in supported_metrics:
        raise ValueError(f"Unsupported simulation metric: {metric}.")

    figure = go.Figure()
    _add_drift_regions(
        figure,
        with_retraining,
    )

    figure.add_trace(
        go.Scatter(
            x=with_retraining["day"],
            y=with_retraining[metric],
            mode="lines",
            name="With retraining",
            line={
                "color": LIFECYCLE_COLORS["with_retraining"],
                "width": 3,
            },
            legendrank=2,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=without_retraining["day"],
            y=without_retraining[metric],
            mode="lines",
            name="Without retraining",
            line={
                "color": LIFECYCLE_COLORS["without_retraining"],
                "width": 3.5,
                "dash": "dash",
            },
            legendrank=1,
        )
    )

    if "event" in with_retraining.columns:
        events = with_retraining.loc[
            with_retraining["event"].fillna("").astype(str).str.strip().ne("")
        ]

        if not events.empty:
            figure.add_trace(
                go.Scatter(
                    x=events["day"],
                    y=events[metric],
                    mode="markers",
                    name="Retraining event",
                    marker={
                        "color": LIFECYCLE_COLORS["retraining_event"],
                        "size": 12,
                        "symbol": "diamond",
                    },
                    text=events["event"].astype(str),
                    hovertemplate=(
                        "Day %{x}<br>"
                        f"{supported_metrics[metric]}: "
                        "%{y:.2f}<br>"
                        "%{text}<extra></extra>"
                    ),
                )
            )
    if "champion_promoted" in with_retraining.columns:
        promotions = with_retraining.loc[with_retraining["champion_promoted"].eq(True)]

        if not promotions.empty:
            figure.add_trace(
                go.Scatter(
                    x=promotions["day"],
                    y=promotions[metric],
                    mode="markers",
                    name="Challenger promoted",
                    marker={
                        "color": LIFECYCLE_COLORS["champion_promoted"],
                        "size": 21,
                        "symbol": "star",
                        "line": {
                            "color": WHITE,
                            "width": 1,
                        },
                    },
                    text=["Challenger promoted to champion"] * len(promotions),
                    hovertemplate=(
                        "Day %{x}<br>"
                        f"{supported_metrics[metric]}: "
                        "%{y:.2f}<br>"
                        "%{text}<extra></extra>"
                    ),
                )
            )
    figure.update_layout(
        height=600,
        hovermode="x unified",
        xaxis_title="Simulation day",
        yaxis_title=supported_metrics[metric],
        legend={
            "orientation": "h",
            "y": 1.15,
            "yanchor": "bottom",
            "x": 0,
            "xanchor": "left",
        },
        margin={
            "l": 75,
            "r": 30,
            "t": 110,
            "b": 65,
        },
    )

    if metric == "bias":
        figure.add_hline(
            y=0.0,
            line_dash="dash",
            line_color=MUTED_GREEN,
        )

    return apply_chart_theme(figure)


def build_segment_chart(
    segment_metrics: pd.DataFrame,
    *,
    metric: str = "rmse",
) -> go.Figure:
    """Compare model variants across evaluation segments."""
    supported_metrics = {
        "rmse": "RMSE",
        "mae": "MAE",
        "wmape_percent": "WMAPE (%)",
        "bias": "Bias",
    }

    if metric not in supported_metrics:
        raise ValueError(f"Unsupported segment metric: {metric}.")

    variant_colors = {
        "Static baseline": REJECTED,
        "Static model": REJECTED,
        "Without retraining": REJECTED,
        "Managed lifecycle": GREEN,
        "With retraining": GREEN,
    }

    figure = go.Figure()

    for model_variant, rows in segment_metrics.groupby(
        "model_variant",
        sort=False,
    ):
        figure.add_trace(
            go.Bar(
                name=str(model_variant),
                x=rows["segment"],
                y=rows[metric],
                marker_color=variant_colors.get(
                    str(model_variant),
                    MUTED_GREEN,
                ),
                text=rows[metric],
                texttemplate=("%{text:.1f}" if metric == "wmape_percent" else "%{text:.0f}"),
                textposition="outside",
                textangle=0,
                textfont={
                    "size": CHART_VALUE_FONT_SIZE,
                    "color": TEXT,
                },
                cliponaxis=False,
                constraintext="none",
            )
        )

    figure.update_layout(
        height=560,
        barmode="group",
        xaxis_title="Store segment",
        yaxis_title=supported_metrics[metric],
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0,
        },
        margin={
            "l": 75,
            "r": 30,
            "t": 80,
            "b": 65,
        },
        uniformtext_minsize=CHART_VALUE_FONT_SIZE,
        uniformtext_mode="show",
    )

    figure.update_xaxes(
        tickvals=[
            "All open stores",
            "Promo stores",
            "Non-promo stores",
        ],
        ticktext=[
            "All stores",
            "Promo",
            "Non-promo",
        ],
    )

    minimum = min(
        0.0,
        float(segment_metrics[metric].min()),
    )
    maximum = max(
        0.0,
        float(segment_metrics[metric].max()),
    )
    padding = max(
        (maximum - minimum) * 0.15,
        1.0,
    )

    figure.update_yaxes(
        range=[
            minimum - padding if minimum < 0 else 0,
            maximum + padding,
        ],
    )

    if metric == "bias":
        figure.add_hline(
            y=0.0,
            line_dash="dash",
            line_color=MUTED_GREEN,
        )

    return apply_chart_theme(figure)
