import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.constants import PROJECT_ROOT
from src.monitoring.costs import build_cost_report


st.set_page_config(page_title="MLOps Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DRIFT_RESULTS_FILE = os.path.join(RESULTS_DIR, "evolution_results.csv")
PERF_RESULTS_FILE = os.path.join(RESULTS_DIR, "performance_demo_history.csv")
PERF_ROLLING_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "monitoring",
    "performance_rolling.parquet",
)


def load_drift_data() -> pd.DataFrame | None:
    if not os.path.exists(DRIFT_RESULTS_FILE):
        return None

    df = pd.read_csv(DRIFT_RESULTS_FILE)

    for col in ["rmse_euro", "static_rmse_euro"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("€", "", regex=False)
                .replace("nan", None)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "drift_detected" in df.columns:
        df["drift_detected"] = (
            df["drift_detected"].astype(str).str.strip().str.lower() == "true"
        )
    else:
        df["drift_detected"] = False

    return df


def load_performance_history() -> pd.DataFrame | None:
    if not os.path.exists(PERF_RESULTS_FILE):
        return None

    df = pd.read_csv(PERF_RESULTS_FILE)

    for col in ["rmse", "mae", "bias", "n_samples", "cumulative_days", "day"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["window_start", "window_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "champion_promoted" in df.columns:
        df["champion_promoted"] = (
            df["champion_promoted"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )
    else:
        df["champion_promoted"] = False

    if "event" not in df.columns:
        df["event"] = None

    return df


def load_performance_rolling() -> pd.DataFrame | None:
    if not os.path.exists(PERF_ROLLING_FILE):
        return None

    df = pd.read_parquet(PERF_ROLLING_FILE)

    for col in ["rmse", "mae", "bias", "n_samples"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["window_start", "window_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def build_monitoring_chart(
    perf_history_df: pd.DataFrame,
    retrain_df: pd.DataFrame,
    promoted_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=perf_history_df["day"],
            y=perf_history_df["rmse"],
            name="RMSE",
            mode="lines+markers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=perf_history_df["day"],
            y=perf_history_df["mae"],
            name="MAE",
            mode="lines+markers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=perf_history_df["day"],
            y=perf_history_df["bias"],
            name="Bias",
            mode="lines+markers",
        )
    )

    if retrain_df is not None and not retrain_df.empty:
        fig.add_trace(
            go.Scatter(
                x=retrain_df["day"],
                y=retrain_df["rmse"],
                mode="markers",
                name="Retrain Triggered",
                marker=dict(size=12, symbol="x"),
            )
        )

    if promoted_df is not None and not promoted_df.empty:
        fig.add_trace(
            go.Scatter(
                x=promoted_df["day"],
                y=promoted_df["rmse"],
                mode="markers",
                name="Champion Promoted",
                marker=dict(size=16, symbol="star"),
            )
        )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        template="plotly_dark",
        xaxis_title="Simulation Day",
        yaxis_title="Metric Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def build_performance_evolution_chart(df_drift: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "static_rmse_euro" in df_drift.columns and df_drift["static_rmse_euro"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_drift["day"],
                y=df_drift["static_rmse_euro"],
                name="Static Model (No Retraining)",
                line=dict(color="#EF553B", width=2, dash="dot"),
                opacity=0.7,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_drift["day"],
            y=df_drift["rmse_euro"],
            name="Adaptive Pipeline (MLOps)",
            line=dict(color="#636EFA", width=3),
            mode="lines+markers",
        )
    )

    if "drift_detected" in df_drift.columns:
        drift_detected = df_drift[df_drift["drift_detected"]]
        stable_days = df_drift[~df_drift["drift_detected"]]

        if not drift_detected.empty:
            fig.add_trace(
                go.Scatter(
                    x=drift_detected["day"],
                    y=drift_detected["rmse_euro"],
                    mode="markers",
                    name="Drift Detected",
                    marker=dict(
                        color="red",
                        size=12,
                        symbol="circle",
                        line=dict(width=2, color="white"),
                    ),
                )
            )

        if not stable_days.empty:
            fig.add_trace(
                go.Scatter(
                    x=stable_days["day"],
                    y=stable_days["rmse_euro"],
                    mode="markers",
                    name="System Stable",
                    marker=dict(color="green", size=8, symbol="circle"),
                )
            )

    fig.add_hline(
        y=1000,
        line_dash="dash",
        line_color="orange",
        annotation_text="Target RMSE Limit",
        annotation_position="top left",
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Simulation Day",
        yaxis_title="RMSE in €",
    )

    return fig


def safe_metric_value(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


# -----------------------------
# Data
# -----------------------------
cost_report = build_cost_report(window_days=7)
perf_history_df = load_performance_history()
perf_roll_df = load_performance_rolling()  # optional, available for future extension
df_drift = load_drift_data()


# -----------------------------
# Layout
# -----------------------------
tab1, tab2 = st.tabs(["Performance", "Costs"])


with tab1:
    st.title("📊 Demand Forecasting - Performance Monitoring")
    st.markdown(
        """
        This dashboard focuses on the production monitoring story of the forecasting system.

        Forecasts are generated first, while actual sales values become available later.
        Once ground truth is available, the system evaluates rolling performance metrics
        and can trigger retraining when degradation persists.
        """
    )

    # ---------------------------------------------------------
    # Primary Story: Performance Demo
    # ---------------------------------------------------------
    st.header("1️⃣ Production Performance Monitoring")
    st.markdown(
        """
        This section is powered by `run_performance_demo.py`.

        It simulates a realistic production lifecycle:
        predictions are logged, delayed ground truth is collected, rolling metrics are
        calculated, and retraining is triggered only when performance degradation persists.
        """
    )

    if perf_history_df is not None and not perf_history_df.empty:
        retrain_df = perf_history_df[perf_history_df["event"] == "retrain"].copy()
        promoted_df = perf_history_df[perf_history_df["champion_promoted"]].copy()
        latest_perf = perf_history_df.iloc[-1]

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Performance Day", int(latest_perf["day"]))
        p2.metric("Rolling RMSE", safe_metric_value(latest_perf.get("rmse")))
        p3.metric("Rolling MAE", safe_metric_value(latest_perf.get("mae")))
        p4.metric("Rolling Bias", safe_metric_value(latest_perf.get("bias")))

        st.subheader("📈 Rolling Metrics Over Time")
        st.plotly_chart(
            build_monitoring_chart(perf_history_df, retrain_df, promoted_df),
            width="stretch",
        )

        st.subheader("🧾 Performance Monitoring History")

        performance_cols = [
            "day",
            "cumulative_days",
            "rmse",
            "mae",
            "bias",
            "n_samples",
            "event",
            "champion_promoted",
        ]

        available_perf_cols = [
            col for col in performance_cols if col in perf_history_df.columns
        ]

        performance_table = perf_history_df[available_perf_cols].tail(15).copy()

        for col in ["rmse", "mae", "bias"]:
            if col in performance_table.columns:
                performance_table[col] = performance_table[col].round(2)

        if "event" in performance_table.columns:
            performance_table["event"] = performance_table["event"].fillna("normal")

        if "champion_promoted" in performance_table.columns:
            performance_table["champion_promoted"] = performance_table["champion_promoted"].map(
                {True: "yes", False: "no"}
            )

        st.dataframe(
            performance_table,
            width="stretch",
            hide_index=True,
        )

        if not retrain_df.empty:
            promoted_count = int(perf_history_df["champion_promoted"].sum())

            if promoted_count > 0:
                st.success(
                    f"Retraining was triggered {len(retrain_df)} time(s). "
                    f"A new champion was promoted {promoted_count} time(s)."
                )
            else:
                st.success(
                    f"Retraining was triggered {len(retrain_df)} time(s) based on monitored performance. "
                    "No challenger was promoted because the existing champion remained stronger."
                )
        else:
            st.info(
                "No retraining has been triggered yet. "
                "This can be a healthy signal if rolling metrics remain within acceptable limits."
            )

    else:
        st.info(
            "No performance demo data found yet. "
            "Run `make demo-forecasting-lifecycle` to generate performance monitoring history."
        )

    # ---------------------------------------------------------
    # Secondary Story: Drift Demo
    # ---------------------------------------------------------
    st.divider()
    st.header("2️⃣ Drift Simulation: Adaptive Pipeline vs. Static Baseline")
    st.markdown(
        """
        This optional section is powered by `run_drift_demo.py`.

        It compares the adaptive MLOps pipeline against a static baseline model and visualizes
        when drift was detected during the simulation.
        """
    )

    if df_drift is not None and not df_drift.empty:
        latest = df_drift.iloc[-1]

        comparison_df = df_drift.dropna(subset=["rmse_euro", "static_rmse_euro"])
        has_static_baseline = (
            "static_rmse_euro" in df_drift.columns
            and df_drift["static_rmse_euro"].notna().any()
            and not comparison_df.empty
        )

        total_saved_error = None
        if has_static_baseline:
            total_saved_error = (
                comparison_df["static_rmse_euro"] - comparison_df["rmse_euro"]
            ).sum()

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Simulation Day", int(latest["day"]))
        d2.metric("Adaptive RMSE", safe_metric_value(latest.get("rmse_euro"), " €"))

        if has_static_baseline:
            latest_static = latest.get("static_rmse_euro")
            baseline_diff = latest.get("rmse_euro") - latest_static
            d3.metric(
                "Static Baseline Delta",
                f"{baseline_diff:.2f} €",
                delta=f"{baseline_diff:.2f} € vs Static",
                delta_color="inverse",
            )
            d4.metric(
                "Total Accuracy Gain",
                f"{total_saved_error:,.2f} €",
                help="Summed RMSE improvement compared with the static baseline.",
            )
        else:
            d3.metric("Static Baseline", "n/a")
            d4.metric(
                "Drift Status",
                "🌩️ Drift" if bool(latest.get("drift_detected")) else "☀️ Stable",
            )

        st.subheader("📈 Adaptive Pipeline Evolution")
        st.plotly_chart(build_performance_evolution_chart(df_drift), width="stretch")

        col_a, col_b = st.columns([2, 1])

        with col_a:
            with st.expander("📝 Recent Drift Simulation Events", expanded=False):
                drift_cols = [
                    "day",
                    "timestamp",
                    "strategy",
                    "drift_detected",
                    "rmse_euro",
                    "static_rmse_euro",
                ]

                available_drift_cols = [
                    col for col in drift_cols if col in df_drift.columns
                ]

                drift_table = df_drift[available_drift_cols].tail(10).copy()

                for col in ["rmse_euro", "static_rmse_euro"]:
                    if col in drift_table.columns:
                        drift_table[col] = drift_table[col].round(2)

                st.dataframe(
                    drift_table,
                    width="stretch",
                    hide_index=True,
                )

        with col_b:
            st.subheader("ℹ️ Drift Demo Notes")
            st.info(
                """
                The drift simulation is a stress scenario.

                It is useful for showing how the system reacts to changing input
                distributions, but the production-oriented monitoring story is the
                rolling performance evaluation shown above.
                """
            )

        if not has_static_baseline:
            st.warning(
                "Static baseline values are not available in the drift results yet. "
                "The adaptive pipeline is shown, but the static comparison line is hidden."
            )

    else:
        st.info(
            "No drift demo data found yet. "
            "Run `uv run python scripts/run_drift_demo.py` or execute it inside the API container "
            "to generate `results/evolution_results.csv`."
        )


with tab2:
    st.title("💰 Cost Monitoring")
    st.markdown(
        """
        This view shows recent training costs and monthly cost scenarios for different
        retraining strategies.
        """
    )

    summary = cost_report["summary"]
    scenarios = cost_report["scenarios"]
    currency = summary["currency"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs (7d)", summary["run_count"])
    c2.metric("Total Cost (7d)", f"{summary['total_training_cost']:.4f} {currency}")
    c3.metric("Avg Cost / Run", f"{summary['avg_training_cost']:.6f} {currency}")
    c4.metric("Avg Duration / Run", f"{summary['avg_training_duration_seconds']:.2f} s")

    st.divider()
    st.subheader("📊 Monthly Cost Scenarios")

    scenario_df = pd.DataFrame(
        [
            {
                "Scenario": name.replace("_", " ").title(),
                "Runs / Month": values["runs_per_month"],
                "Estimated Monthly Cost": values["estimated_monthly_cost"],
            }
            for name, values in scenarios.items()
        ]
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.dataframe(
            scenario_df.style.format(
                {
                    "Estimated Monthly Cost": lambda x: f"{x:.4f} {currency}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with right:
        chart_df = scenario_df.set_index("Scenario")[["Estimated Monthly Cost"]]
        st.bar_chart(chart_df, width="stretch")

    st.divider()
    st.caption(
        "Cost estimates are based on a configured hourly rate and are intended as an "
        "approximation of training costs, not as a complete cloud billing report."
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("System Health")
st.sidebar.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.markdown(
    """
**Automated Stack:**
- [x] Forecast performance monitoring
- [x] Delayed ground truth evaluation
- [x] Drift detection
- [x] MLflow Registry
- [x] Prefect Orchestration
- [x] Feature State Persistence
"""
)