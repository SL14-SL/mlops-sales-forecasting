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
DRIFT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DRIFT_RESULTS_FILE = os.path.join(DRIFT_RESULTS_DIR, "evolution_results_90days_baseline.csv")
PERF_RESULTS_FILE = os.path.join(DRIFT_RESULTS_DIR, "performance_demo_history.csv")
PERF_ROLLING_FILE = os.path.join(PROJECT_ROOT, "data", "monitoring", "performance_rolling.parquet")


def load_drift_data() -> pd.DataFrame | None:
    if not os.path.exists(DRIFT_RESULTS_FILE):
        print(f"Warning: File not found at {DRIFT_RESULTS_FILE}")
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


def build_performance_evolution_chart(df_drift: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_drift["day"],
            y=df_drift["static_rmse_euro"],
            name="Static Model (No Retraining)",
            line=dict(color="#EF553B", width=2, dash="dot"),
            opacity=0.5,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_drift["day"],
            y=df_drift["rmse_euro"],
            name="Adaptive Pipeline (MLOps)",
            line=dict(color="#636EFA", width=3),
            mode="lines",
        )
    )

    drift_detected = df_drift[df_drift["drift_detected"]]
    stable_days = df_drift[~df_drift["drift_detected"]]

    fig.add_trace(
        go.Scatter(
            x=drift_detected["day"],
            y=drift_detected["rmse_euro"],
            mode="markers",
            name="Emergency Retraining",
            marker=dict(
                color="red",
                size=12,
                symbol="circle",
                line=dict(width=2, color="white"),
            ),
        )
    )

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
        annotation_text="Target Limit (SLA)",
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

    # Retraining wurde ausgelöst
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

    # Neuer Champion wurde wirklich promoted
    if promoted_df is not None and not promoted_df.empty:
        fig.add_trace(
            go.Scatter(
                x=promoted_df["day"],
                y=promoted_df["rmse"],
                mode="markers",
                name="Model deployed",
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


# -----------------------------
# Data
# -----------------------------
cost_report = build_cost_report(window_days=7)
df_drift = load_drift_data()
perf_history_df = load_performance_history()
perf_roll_df = load_performance_rolling()  # Optional geladen, falls später gebraucht


# -----------------------------
# Layout
# -----------------------------
tab1, tab2 = st.tabs(["Performance", "Costs"])


with tab1:
    st.title("🛡️ Demand Forecasting - Adaptive Monitoring")
    st.markdown(
        """
        Dieses Dashboard zeigt die **automatisierte Selbsterhaltung** des Modells.
        Es vergleicht unsere adaptive Pipeline mit einem statischen Modell (ohne MLOps).
        """
    )

    if df_drift is not None and not df_drift.empty:
        latest = df_drift.iloc[-1]

        comparison_df = df_drift.dropna(subset=["rmse_euro", "static_rmse_euro"])
        total_saved_error = (comparison_df["static_rmse_euro"] - comparison_df["rmse_euro"]).sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simulation Day", int(latest["day"]))

        baseline_diff = latest["rmse_euro"] - latest["static_rmse_euro"]
        m2.metric(
            "Current RMSE",
            f"{latest['rmse_euro']:.2f} €",
            delta=f"{baseline_diff:.2f} € vs Static",
            delta_color="inverse",
        )

        m3.metric(
            "Total Accuracy Gain",
            f"{total_saved_error:,.2f} €",
            help="Summierte RMSE-Ersparnis gegenüber dem statischen Modell",
        )

        status_icon = "🌩️ Drift" if latest["drift_detected"] else "☀️ Stable"
        m4.metric("System Status", status_icon)

        st.divider()
        st.subheader("📈 Performance Evolution: Adaptive vs. Static Baseline")
        st.plotly_chart(build_performance_evolution_chart(df_drift), width="stretch")

        st.divider()
        st.header("📊 Model Performance Monitoring")

        if perf_history_df is not None and not perf_history_df.empty:
            retrain_df = perf_history_df[perf_history_df["event"] == "retrain"].copy()
            promoted_df = perf_history_df[perf_history_df["champion_promoted"]]
            latest_perf = perf_history_df.iloc[-1]

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Performance Day", int(latest_perf["day"]))
            p2.metric("Rolling RMSE", f"{latest_perf['rmse']:.2f}")
            p3.metric("Rolling MAE", f"{latest_perf['mae']:.2f}")
            p4.metric("Rolling Bias", f"{latest_perf['bias']:.2f}")

            st.subheader("📈 Rolling Metrics Over Time")
            st.plotly_chart(
                build_monitoring_chart(perf_history_df, retrain_df, promoted_df),
                width="stretch",
            )

            st.subheader("🧾 Performance Monitoring History")
            st.dataframe(
                perf_history_df.tail(15),
                width="stretch",
            )
        else:
            st.info("No performance demo data found yet. Run `python scripts/run_performance_demo.py` first.")

        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.subheader("📝 Recent System Events")
            log_df = df_drift[
                ["day", "timestamp", "strategy", "drift_detected", "rmse_euro", "static_rmse_euro"]
            ].tail(10).copy()

            st.dataframe(
                log_df.style.highlight_between(
                    left=1000,
                    right=5000,
                    subset=["rmse_euro"],
                    color="#992222",
                ),
                width="stretch",
            )

        with col_b:
            st.subheader("ℹ️ Strategy Info")
            st.info(
                f"""
                **Current Strategy:** {latest['strategy']}

                Die **rote gestrichelte Linie** zeigt das Modell Version 1.
                Man sieht deutlich, dass es den Umsatz-Shift (Tag 6) nicht verkraftet.

                Die **Markierungspunkte** auf der blauen Linie zeigen, wann der K-S Test
                einen Drift erkannt und ein Retraining erzwungen hat.
                """
            )
    else:
        st.warning("No evolution data found. Please run `run_drift_demo.py` first.")


with tab2:
    st.title("💰 Cost Monitoring")
    st.markdown(
        """
        Diese Ansicht zeigt die **Trainingskosten** der letzten Tage sowie
        **monatliche Kostenszenarien** für verschiedene Retraining-Strategien.
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
        "Hinweis: Die Kostenschätzung basiert aktuell auf einer konfigurierten "
        "Stundenrate und dient als Näherung für Trainingskosten, nicht als "
        "vollständige Cloud-Billing-Abrechnung."
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("System Health")
st.sidebar.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.markdown(
    """
**Automated Stack:**
- [x] Drift: K-S Test (Sales)
- [x] MLflow Registry (V1 vs Champion)
- [x] Prefect Orchestration
- [x] Feature State Persistence
"""
)