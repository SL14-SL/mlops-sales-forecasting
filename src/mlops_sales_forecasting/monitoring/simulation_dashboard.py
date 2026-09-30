from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

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

    figure.add_trace(
        go.Scatter(
            x=with_retraining["day"],
            y=with_retraining[metric],
            mode="lines",
            name="With retraining",
            line={
                "color": "#00CC96",
                "width": 2.5,
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
                "color": "#EF553B",
                "width": 3,
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
                        "color": "#AB63FA",
                        "size": 9,
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
                        "color": "#FFA15A",
                        "size": 17,
                        "symbol": "star",
                        "line": {
                            "color": "#FFFFFF",
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
        height=480,
        hovermode="x unified",
        xaxis_title="Simulation day",
        yaxis_title=supported_metrics[metric],
        legend={
            "orientation": "h",
            "y": 1.12,
        },
        margin={
            "l": 40,
            "r": 20,
            "t": 30,
            "b": 40,
        },
    )

    if metric == "bias":
        figure.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="#7F7F7F",
        )

    return figure


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
            )
        )

    figure.update_layout(
        height=460,
        barmode="group",
        xaxis_title="Store segment",
        yaxis_title=supported_metrics[metric],
        legend={
            "orientation": "h",
            "y": 1.12,
        },
        margin={
            "l": 40,
            "r": 20,
            "t": 30,
            "b": 40,
        },
    )

    if metric == "bias":
        figure.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="#7F7F7F",
        )

    return figure
