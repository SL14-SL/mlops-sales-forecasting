from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.constants import PROJECT_ROOT
from src.monitoring.costs import build_cost_report


st.set_page_config(
    page_title="MLOps Dashboard",
    layout="wide",
)


RESULTS_DIR = PROJECT_ROOT / "results"
PERFORMANCE_ROLLING_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "performance_rolling.parquet"
)

REQUIRED_LIFECYCLE_COLUMNS = {
    "day",
    "rmse",
    "mae",
    "bias",
}


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Normalize common boolean representations."""
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def normalize_lifecycle_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize one lifecycle result table for dashboard rendering."""
    frame = frame.copy()

    for column in [
        "day",
        "cumulative_days",
        "rmse",
        "mae",
        "bias",
        "n_samples",
        "drift_start_day",
        "drift_duration_days",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    for column in [
        "window_start",
        "window_end",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
            )

    if "event" not in frame.columns:
        frame["event"] = None

    if "champion_promoted" in frame.columns:
        frame["champion_promoted"] = (
            parse_boolean_series(
                frame["champion_promoted"]
            )
        )
    else:
        frame["champion_promoted"] = False

    if "retraining_enabled" in frame.columns:
        frame["retraining_enabled"] = (
            parse_boolean_series(
                frame["retraining_enabled"]
            )
        )

    if "day" not in frame.columns:
        frame["day"] = range(
            1,
            len(frame) + 1,
        )

    return (
        frame
        .dropna(subset=["day", "rmse"])
        .sort_values("day")
        .reset_index(drop=True)
    )


def is_lifecycle_result(path: Path) -> bool:
    """Return whether a CSV has the required lifecycle columns."""
    try:
        columns = set(
            pd.read_csv(
                path,
                nrows=0,
            ).columns
        )
    except (OSError, pd.errors.ParserError):
        return False

    return REQUIRED_LIFECYCLE_COLUMNS.issubset(
        columns
    )


def discover_lifecycle_results() -> list[Path]:
    """Discover compatible lifecycle CSV files, newest first."""
    if not RESULTS_DIR.exists():
        return []

    candidates = [
        path
        for path in RESULTS_DIR.glob("*.csv")
        if is_lifecycle_result(path)
    ]

    return sorted(
        candidates,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_lifecycle_result(
    path: Path,
) -> pd.DataFrame | None:
    """Load one lifecycle result while tolerating a file being updated."""
    try:
        frame = pd.read_csv(path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return None

    if not REQUIRED_LIFECYCLE_COLUMNS.issubset(
        frame.columns
    ):
        return None

    return normalize_lifecycle_frame(frame)


def load_rolling_fallback() -> pd.DataFrame | None:
    """Load native monitoring metrics when no lifecycle CSV exists."""
    if not PERFORMANCE_ROLLING_FILE.exists():
        return None

    try:
        frame = pd.read_parquet(
            PERFORMANCE_ROLLING_FILE
        )
    except (OSError, ValueError):
        return None

    if not {"rmse", "mae", "bias"}.issubset(
        frame.columns
    ):
        return None

    frame = frame.copy()
    frame["day"] = range(
        1,
        len(frame) + 1,
    )
    frame["event"] = None
    frame["champion_promoted"] = False

    return normalize_lifecycle_frame(frame)


def safe_metric_value(
    value: float | int | None,
) -> str:
    """Format dashboard metric values."""
    if value is None or pd.isna(value):
        return "n/a"

    return f"{float(value):.2f}"


def describe_run(
    path: Path,
    frame: pd.DataFrame,
) -> str:
    """Create a readable label for one lifecycle result."""
    scenario = "unknown"
    retraining = "unknown"

    if "scenario" in frame.columns:
        values = frame["scenario"].dropna()
        if not values.empty:
            scenario = str(values.iloc[-1])

    if "retraining_enabled" in frame.columns:
        enabled = bool(
            frame["retraining_enabled"].iloc[-1]
        )
        retraining = (
            "retraining enabled"
            if enabled
            else "retraining disabled"
        )

    return (
        f"{path.name} | {scenario} | {retraining} | "
        f"{len(frame)} days"
    )


def build_monitoring_chart(
    history: pd.DataFrame,
) -> go.Figure:
    """Build the operational rolling-metrics chart."""
    figure = go.Figure()

    colors = {
        "rmse": "#636EFA",
        "mae": "#EF553B",
        "bias": "#00CC96",
    }

    for column, label in [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("bias", "Bias"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=history["day"],
                y=history[column],
                name=label,
                mode="lines+markers",
                line={
                    "color": colors[column],
                    "width": 2.5,
                },
            )
        )

    retrain_rows = history.loc[
        history["event"].eq("retrain")
    ]

    if not retrain_rows.empty:
        figure.add_trace(
            go.Scatter(
                x=retrain_rows["day"],
                y=retrain_rows["rmse"],
                name="Retraining triggered",
                mode="markers",
                marker={
                    "size": 14,
                    "symbol": "x",
                    "color": "#AB63FA",
                },
            )
        )

    promoted_rows = history.loc[
        history["champion_promoted"]
    ]

    if not promoted_rows.empty:
        figure.add_trace(
            go.Scatter(
                x=promoted_rows["day"],
                y=promoted_rows["rmse"],
                name="Final-refit champion promoted",
                mode="markers",
                marker={
                    "size": 17,
                    "symbol": "star",
                    "color": "#FFA15A",
                },
            )
        )

    if {
        "drift_start_day",
        "drift_duration_days",
    }.issubset(history.columns):
        start_values = history[
            "drift_start_day"
        ].dropna()
        duration_values = history[
            "drift_duration_days"
        ].dropna()

        if not start_values.empty and not duration_values.empty:
            drift_start = int(start_values.iloc[-1])
            full_drift = (
                drift_start
                + int(duration_values.iloc[-1])
                - 1
            )

            figure.add_vrect(
                x0=drift_start,
                x1=full_drift,
                fillcolor="#F2C14E",
                opacity=0.16,
                line_width=0,
                annotation_text="Drift ramp-up",
                annotation_position="top left",
            )

    figure.update_layout(
        height=540,
        hovermode="x unified",
        template="plotly_dark",
        xaxis_title="Simulation day",
        yaxis_title="Metric value",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figure


def build_rmse_comparison_chart(
    without_retraining: pd.DataFrame,
    with_retraining: pd.DataFrame,
) -> go.Figure:
    """Compare rolling RMSE for matching simulation days."""
    comparison = (
        without_retraining[
            ["day", "rmse"]
        ]
        .merge(
            with_retraining[
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
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=comparison["day"],
            y=comparison["rmse_without"],
            name="Without retraining",
            mode="lines",
            line={
                "color": "#EF553B",
                "width": 2.5,
                "dash": "dash",
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=comparison["day"],
            y=comparison["rmse_with"],
            name="With retraining",
            mode="lines",
            line={
                "color": "#636EFA",
                "width": 3,
            },
        )
    )

    retrain_rows = with_retraining.loc[
        with_retraining["event"].eq("retrain")
    ]

    if not retrain_rows.empty:
        retrain_day = int(
            retrain_rows.iloc[0]["day"]
        )
        figure.add_vline(
            x=retrain_day,
            line_dash="dash",
            line_color="#00CC96",
            annotation_text="Retraining",
            annotation_position="top right",
        )

    figure.update_layout(
        height=500,
        hovermode="x unified",
        template="plotly_dark",
        xaxis_title="Simulation day",
        yaxis_title="Rolling RMSE",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figure


def get_retraining_enabled(
    frame: pd.DataFrame,
) -> bool | None:
    """Return the retraining mode stored in one lifecycle result."""
    if "retraining_enabled" not in frame.columns:
        return None

    values = frame["retraining_enabled"].dropna()

    if values.empty:
        return None

    return bool(values.iloc[-1])


cost_report = build_cost_report(window_days=7)
lifecycle_paths = discover_lifecycle_results()
lifecycle_runs = {
    path: load_lifecycle_result(path)
    for path in lifecycle_paths
}
lifecycle_runs = {
    path: frame
    for path, frame in lifecycle_runs.items()
    if frame is not None and not frame.empty
}


tab_performance, tab_costs = st.tabs(
    [
        "Performance",
        "Costs",
    ]
)


with tab_performance:
    st.title(
        "📊 Demand Forecasting - Performance Monitoring"
    )
    st.markdown(
        """
        This dashboard reads lifecycle result files directly from `results/`.
        Forecasts are generated first, delayed ground truth becomes available
        later, and rolling performance metrics are updated throughout the
        simulation. No manual result-file copy is required.
        """
    )

    st.header(
        "Production Performance Monitoring"
    )

    if lifecycle_runs:
        available_paths = list(
            lifecycle_runs.keys()
        )

        selected_path = st.selectbox(
            "Lifecycle run",
            options=available_paths,
            format_func=lambda path: describe_run(
                path,
                lifecycle_runs[path],
            ),
        )

        selected_history = lifecycle_runs[
            selected_path
        ]
    else:
        selected_path = None
        selected_history = load_rolling_fallback()

    if selected_history is not None and not selected_history.empty:
        latest = selected_history.iloc[-1]

        metric_day, metric_rmse, metric_mae, metric_bias = (
            st.columns(4)
        )

        metric_day.metric(
            "Performance day",
            int(latest["day"]),
        )
        metric_rmse.metric(
            "Rolling RMSE",
            safe_metric_value(
                latest.get("rmse")
            ),
        )
        metric_mae.metric(
            "Rolling MAE",
            safe_metric_value(
                latest.get("mae")
            ),
        )
        metric_bias.metric(
            "Rolling bias",
            safe_metric_value(
                latest.get("bias")
            ),
        )

        st.subheader(
            "📈 Rolling Metrics Over Time"
        )
        st.plotly_chart(
            build_monitoring_chart(
                selected_history
            ),
            width="stretch",
        )

        retrain_rows = selected_history.loc[
            selected_history["event"].eq(
                "retrain"
            )
        ]
        promotion_count = int(
            selected_history[
                "champion_promoted"
            ].sum()
        )

        if not retrain_rows.empty:
            st.success(
                f"Retraining was triggered "
                f"{len(retrain_rows)} time(s). "
                f"A final-refit champion was promoted "
                f"{promotion_count} time(s)."
            )
        else:
            st.info(
                "No retraining event is stored in "
                "the selected lifecycle run."
            )

        with st.expander(
            "🧾 Recent monitoring history",
            expanded=False,
        ):
            table_columns = [
                column
                for column in [
                    "day",
                    "cumulative_days",
                    "rmse",
                    "mae",
                    "bias",
                    "n_samples",
                    "event",
                    "champion_promoted",
                    "scenario",
                    "retraining_enabled",
                ]
                if column in selected_history.columns
            ]

            table = selected_history[
                table_columns
            ].tail(20).copy()

            for column in [
                "rmse",
                "mae",
                "bias",
            ]:
                if column in table.columns:
                    table[column] = table[
                        column
                    ].round(2)

            if "event" in table.columns:
                table["event"] = table[
                    "event"
                ].fillna("normal")

            st.dataframe(
                table,
                width="stretch",
                hide_index=True,
            )
    else:
        st.info(
            "No lifecycle CSV or rolling monitoring "
            "artifact is available yet. Run "
            "`run_performance_demo.py` to generate data."
        )

    enabled_runs = {
        path: frame
        for path, frame in lifecycle_runs.items()
        if get_retraining_enabled(frame) is True
    }
    disabled_runs = {
        path: frame
        for path, frame in lifecycle_runs.items()
        if get_retraining_enabled(frame) is False
    }

    st.divider()
    st.header(
        "Controlled Retraining Comparison"
    )
    st.markdown(
        """
        Select matching lifecycle runs to compare the adaptive pipeline with a
        static no-retraining baseline. For a fair comparison, both files should
        use the same scenario, drift parameters, initial champion and ground
        truth.
        """
    )

    if enabled_runs and disabled_runs:
        comparison_left, comparison_right = (
            st.columns(2)
        )

        with comparison_left:
            without_path = st.selectbox(
                "Without retraining",
                options=list(disabled_runs.keys()),
                format_func=lambda path: describe_run(
                    path,
                    disabled_runs[path],
                ),
            )

        with comparison_right:
            with_path = st.selectbox(
                "With retraining",
                options=list(enabled_runs.keys()),
                format_func=lambda path: describe_run(
                    path,
                    enabled_runs[path],
                ),
            )

        without_history = disabled_runs[
            without_path
        ]
        with_history = enabled_runs[
            with_path
        ]

        st.plotly_chart(
            build_rmse_comparison_chart(
                without_history,
                with_history,
            ),
            width="stretch",
        )

        st.caption(
            "This chart compares rolling RMSE. The segment-level "
            "RMSE, MAE, WMAPE and bias report remains the final "
            "offline model-quality evaluation."
        )
    else:
        st.info(
            "Both a retraining-enabled and a retraining-disabled "
            "lifecycle CSV are required for the comparison."
        )


with tab_costs:
    st.title(
        "💰 Cost Monitoring"
    )
    st.markdown(
        """
        This view shows recent training costs and monthly cost scenarios for
        different retraining strategies.
        """
    )

    summary = cost_report["summary"]
    scenarios = cost_report["scenarios"]
    currency = summary["currency"]

    cost_1, cost_2, cost_3, cost_4 = (
        st.columns(4)
    )

    cost_1.metric(
        "Runs (7d)",
        summary["run_count"],
    )
    cost_2.metric(
        "Total cost (7d)",
        f"{summary['total_training_cost']:.4f} "
        f"{currency}",
    )
    cost_3.metric(
        "Average cost per run",
        f"{summary['avg_training_cost']:.6f} "
        f"{currency}",
    )
    cost_4.metric(
        "Average duration per run",
        f"{summary['avg_training_duration_seconds']:.2f} s",
    )

    st.divider()
    st.subheader(
        "📊 Monthly Cost Scenarios"
    )

    scenario_frame = pd.DataFrame(
        [
            {
                "Scenario": name.replace(
                    "_",
                    " ",
                ).title(),
                "Runs / Month": values[
                    "runs_per_month"
                ],
                "Estimated Monthly Cost": values[
                    "estimated_monthly_cost"
                ],
            }
            for name, values in scenarios.items()
        ]
    )

    cost_table_column, cost_chart_column = (
        st.columns([1.2, 1])
    )

    with cost_table_column:
        st.dataframe(
            scenario_frame.style.format(
                {
                    "Estimated Monthly Cost": (
                        lambda value: (
                            f"{value:.4f} {currency}"
                        )
                    ),
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with cost_chart_column:
        st.bar_chart(
            scenario_frame.set_index(
                "Scenario"
            )[
                ["Estimated Monthly Cost"]
            ],
            width="stretch",
        )

    st.divider()
    st.caption(
        "Cost estimates use the configured hourly rate and "
        "are not a complete cloud billing report."
    )


st.sidebar.header(
    "System Health"
)
st.sidebar.info(
    "Last update: "
    f"{datetime.now().strftime('%H:%M:%S')}"
)
st.sidebar.metric(
    "Discovered lifecycle runs",
    len(lifecycle_runs),
)
st.sidebar.markdown(
    """
    **Automated Stack:**

    - [x] Forecast performance monitoring
    - [x] Delayed ground-truth evaluation
    - [x] Controlled retraining comparison
    - [x] MLflow Registry
    - [x] Prefect orchestration
    - [x] Feature-state persistence
    """
)