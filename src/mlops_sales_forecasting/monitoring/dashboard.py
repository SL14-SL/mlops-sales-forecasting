import json
import os
from collections.abc import Mapping
from typing import Any
from urllib.request import urlopen

import fsspec
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mlops_sales_forecasting.configs.loader import (
    load_config,
)
from mlops_sales_forecasting.configs.paths import (
    join_uri,
)
from mlops_sales_forecasting.monitoring.costs import (
    build_training_cost_report,
)
from mlops_sales_forecasting.storage.filesystem import (
    file_exists,
)


def load_monitoring_summary(
    api_base_url: str,
) -> dict[str, Any]:
    """Load the operational summary from the model API."""
    url = f"{api_base_url.rstrip('/')}/monitoring/summary"

    with urlopen(
        url,
        timeout=5,
    ) as response:
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise ValueError("Monitoring summary must be a JSON object.")

    return payload


def load_parquet_if_available(
    path: str,
) -> pd.DataFrame:
    """Load a local or remote Parquet file when available."""
    if not file_exists(path):
        return pd.DataFrame()

    try:
        with fsspec.open(
            path,
            "rb",
        ) as file:
            return pd.read_parquet(file)
    except (
        OSError,
        TypeError,
        ValueError,
    ):
        return pd.DataFrame()


def build_monitoring_paths(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Build persisted monitoring-history paths."""
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a valid 'paths' section.")

    monitoring_path = paths.get("monitoring")

    if not isinstance(monitoring_path, str) or not monitoring_path:
        raise ValueError("Config must contain a non-empty 'paths.monitoring' value.")

    return {
        "performance": join_uri(
            monitoring_path,
            "performance_rolling.parquet",
        ),
        "feature_drift": join_uri(
            monitoring_path,
            "feature_drift_history.parquet",
        ),
    }


def build_performance_chart(
    history: pd.DataFrame,
) -> go.Figure:
    """Build a rolling forecast-performance chart."""
    required_columns = {
        "window_end",
        "rmse",
        "mae",
        "bias",
    }

    if not required_columns.issubset(history.columns):
        raise ValueError("Performance history does not contain the required columns.")

    normalized = history.copy()
    normalized["window_end"] = pd.to_datetime(
        normalized["window_end"],
        errors="coerce",
        utc=True,
    )
    normalized = normalized.dropna(
        subset=[
            "window_end",
        ]
    ).sort_values("window_end")

    figure = go.Figure()

    for column, label, color in (
        ("rmse", "RMSE", "#636EFA"),
        ("mae", "MAE", "#EF553B"),
        ("bias", "Bias", "#00CC96"),
    ):
        figure.add_trace(
            go.Scatter(
                x=normalized["window_end"],
                y=normalized[column],
                mode="lines+markers",
                name=label,
                line={
                    "color": color,
                    "width": 2.5,
                },
            )
        )

    figure.update_layout(
        height=450,
        hovermode="x unified",
        xaxis_title="Evaluation window",
        yaxis_title="Metric value",
        legend={
            "orientation": "h",
            "y": 1.1,
        },
    )

    return figure


def build_drift_chart(
    history: pd.DataFrame,
) -> go.Figure:
    """Build a chart for the latest feature-drift scores."""
    required_columns = {
        "timestamp",
        "feature",
        "score",
        "drift_detected",
    }

    if not required_columns.issubset(history.columns):
        raise ValueError("Feature-drift history does not contain the required columns.")

    normalized = history.copy()
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"],
        errors="coerce",
        utc=True,
    )
    normalized = normalized.dropna(
        subset=[
            "timestamp",
        ]
    )

    if normalized.empty:
        raise ValueError("Feature-drift history contains no valid timestamps.")

    latest_timestamp = normalized["timestamp"].max()
    latest = normalized.loc[normalized["timestamp"] == latest_timestamp].sort_values(
        "score",
        ascending=False,
    )

    colors = [("#EF553B" if bool(drifted) else "#00CC96") for drifted in latest["drift_detected"]]

    figure = go.Figure(
        data=[
            go.Bar(
                x=latest["feature"],
                y=latest["score"],
                marker_color=colors,
                name="Drift score",
            )
        ]
    )
    figure.update_layout(
        height=420,
        xaxis_title="Feature",
        yaxis_title="Drift score",
    )

    return figure


def _metric_value(
    value: Any,
    *,
    decimals: int = 2,
) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.{decimals}f}"
    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def _render_serving_status(
    summary: Mapping[str, Any],
) -> None:
    serving = summary.get("serving", {})

    if not isinstance(serving, Mapping):
        serving = {}

    ready = bool(
        serving.get(
            "ready",
            False,
        )
    )

    first, second, third = st.columns(3)
    first.metric(
        "Serving status",
        "Ready" if ready else "Not ready",
    )
    second.metric(
        "Active release",
        serving.get("active_release_id") or "none",
    )
    third.metric(
        "Summary generated",
        summary.get("generated_at_utc") or "unknown",
    )

    reload_error = serving.get("last_reload_error")

    if reload_error:
        st.warning(str(reload_error))


def _render_current_monitoring(
    summary: Mapping[str, Any],
) -> None:
    performance = summary.get("performance", {})
    drift = summary.get("feature_drift", {})
    retraining = summary.get("retraining", {})

    if not isinstance(performance, Mapping):
        performance = {}

    if not isinstance(drift, Mapping):
        drift = {}

    if not isinstance(retraining, Mapping):
        retraining = {}

    st.subheader("Current monitoring state")

    rmse, mae, bias, samples = st.columns(4)
    rmse.metric(
        "RMSE",
        _metric_value(performance.get("rmse")),
    )
    mae.metric(
        "MAE",
        _metric_value(performance.get("mae")),
    )
    bias.metric(
        "Bias",
        _metric_value(performance.get("bias")),
    )
    samples.metric(
        "Matched samples",
        performance.get("n_samples") or 0,
    )

    drifted_features = drift.get(
        "drifted_feature_names",
        [],
    )
    st.metric(
        "Drifted features",
        drift.get("drifted_features") or 0,
    )

    if drifted_features:
        st.warning(
            "Persistent or current drift: "
            + ", ".join(str(feature) for feature in drifted_features)
        )

    st.write(
        {
            "retraining_action": retraining.get("action"),
            "trigger_types": retraining.get(
                "trigger_types",
                [],
            ),
            "last_retrained_at_utc": retraining.get("last_retrained_at_utc"),
            "candidate_run_id": retraining.get("candidate_run_id"),
            "champion_promoted": retraining.get(
                "champion_promoted",
                False,
            ),
        }
    )


def _render_cost_report(
    report: Mapping[str, Any],
) -> None:
    st.subheader("Estimated training costs")

    if not report.get(
        "enabled",
        False,
    ):
        st.info("Training-cost estimation is disabled.")
        return

    summary = report.get("summary", {})
    scenarios = report.get("scenarios", {})

    if not isinstance(summary, Mapping):
        summary = {}

    if not isinstance(scenarios, Mapping):
        scenarios = {}

    currency = str(
        summary.get(
            "currency",
            "EUR",
        )
    )

    run_count, total_cost, average_cost = st.columns(3)
    run_count.metric(
        "Completed runs",
        summary.get(
            "run_count",
            0,
        ),
    )
    total_cost.metric(
        "Observed-window estimate",
        (f"{_metric_value(summary.get('total_estimated_cost'), decimals=4)} {currency}"),
    )
    average_cost.metric(
        "Average per run",
        (f"{_metric_value(summary.get('average_estimated_cost'), decimals=4)} {currency}"),
    )

    scenario_rows = []

    for name, values in scenarios.items():
        if not isinstance(values, Mapping):
            continue

        scenario_rows.append(
            {
                "scenario": name,
                "runs_per_month": values.get(
                    "runs_per_month",
                    0,
                ),
                "estimated_monthly_cost": (
                    values.get(
                        "estimated_monthly_cost",
                        0.0,
                    )
                ),
                "currency": currency,
            }
        )

    if scenario_rows:
        st.dataframe(
            pd.DataFrame(scenario_rows),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Cost values are estimates based on observed "
        "training duration and the configured hourly rate."
    )


def render_dashboard() -> None:
    """Render the forecasting operations dashboard."""
    st.set_page_config(
        page_title=("Sales Forecasting Operations"),
        page_icon="📈",
        layout="wide",
    )

    st.title("Sales Forecasting Operations")
    st.caption("Serving, model performance, drift, retraining and estimated training costs.")

    config = load_config()
    api_base_url = os.getenv(
        "DASHBOARD_API_URL",
        "http://localhost:8000",
    )

    try:
        summary = load_monitoring_summary(api_base_url)
    except Exception as error:
        st.error(f"Could not load the monitoring summary: {error}")
        summary = {}

    _render_serving_status(summary)
    _render_current_monitoring(summary)

    paths = build_monitoring_paths(config)
    performance_history = load_parquet_if_available(paths["performance"])
    drift_history = load_parquet_if_available(paths["feature_drift"])

    st.subheader("Forecast performance history")

    if performance_history.empty:
        st.info("No performance history is available yet.")
    else:
        try:
            st.plotly_chart(
                build_performance_chart(performance_history),
                use_container_width=True,
            )
        except ValueError as error:
            st.warning(str(error))

    st.subheader("Latest feature drift")

    if drift_history.empty:
        st.info("No feature-drift history is available yet.")
    else:
        try:
            st.plotly_chart(
                build_drift_chart(drift_history),
                use_container_width=True,
            )
        except ValueError as error:
            st.warning(str(error))

    try:
        cost_report = build_training_cost_report(config)
        _render_cost_report(cost_report)
    except Exception as error:
        st.subheader("Estimated training costs")
        st.info(f"Training-cost information is currently unavailable: {error}")

    st.subheader("Operational tools")
    st.markdown(
        "- [Grafana](http://localhost:3000)\n"
        "- [Prometheus](http://localhost:9090)\n"
        "- [MLflow](http://localhost:5000)\n"
        "- [Prefect](http://localhost:4200)"
    )


if __name__ == "__main__":
    render_dashboard()
