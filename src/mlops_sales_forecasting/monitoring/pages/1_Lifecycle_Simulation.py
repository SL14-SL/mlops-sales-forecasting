import os
from pathlib import Path

import streamlit as st

from mlops_sales_forecasting.monitoring.simulation_dashboard import (
    build_lifecycle_metric_chart,
    build_segment_chart,
    build_simulation_summary,
    load_reference_simulation,
)

DEFAULT_RESULTS_PATH = Path("examples/lifecycle_simulation")


def _results_path() -> Path:
    return Path(
        os.getenv(
            "SIMULATION_RESULTS_PATH",
            str(DEFAULT_RESULTS_PATH),
        )
    )


def main() -> None:
    """Render the lifecycle-simulation dashboard."""
    st.set_page_config(
        page_title="Lifecycle Simulation",
        page_icon="🔁",
        layout="wide",
    )

    st.title("Rossmann Lifecycle Simulation")
    st.caption(
        "Comparison of a static forecasting model with the drift-aware retraining lifecycle."
    )

    results_path = _results_path()

    try:
        (
            without_retraining,
            with_retraining,
            segment_metrics,
        ) = load_reference_simulation(results_path)
    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        st.error("Lifecycle simulation results could not be loaded.")
        st.code(str(error))
        st.stop()

    summary = build_simulation_summary(
        without_retraining,
        with_retraining,
    )

    st.subheader("Outcome")

    (
        final_without,
        final_with,
        improvement,
        retraining_events,
        promotions,
    ) = st.columns(5)

    final_without.metric(
        "Final RMSE: static",
        f"{summary['final_rmse_without_retraining']:.2f}",
    )
    final_with.metric(
        "Final RMSE: retrained",
        f"{summary['final_rmse_with_retraining']:.2f}",
    )
    improvement.metric(
        "RMSE improvement",
        f"{summary['relative_rmse_improvement']:.1%}",
    )
    retraining_events.metric(
        "Retraining events",
        summary["retraining_events"],
    )
    promotions.metric(
        "Promotions",
        summary["promotion_events"],
    )

    st.info(
        "The simulated promotion-aware lifecycle reduces the "
        "final RMSE by "
        f"{summary['relative_rmse_improvement']:.1%} "
        "relative to the static-model scenario."
    )

    st.subheader("Performance over the simulation")

    selected_metric = st.selectbox(
        "Lifecycle metric",
        options=[
            "rmse",
            "mae",
            "bias",
        ],
        format_func={
            "rmse": "RMSE",
            "mae": "MAE",
            "bias": "Bias",
        }.get,
    )

    lifecycle_figure = build_lifecycle_metric_chart(
        without_retraining,
        with_retraining,
        metric=selected_metric,
    )
    st.plotly_chart(
        lifecycle_figure,
        use_container_width=True,
    )

    with st.expander("How to read the lifecycle comparison"):
        st.markdown(
            """
- **Without retraining** keeps the original model active while
  the promotion behavior changes.
- **With retraining** evaluates monitoring signals, trains
  candidates and promotes an approved replacement.
- Purple diamonds mark completed retraining events.
- The orange star marks the challenger that passed evaluation and was promoted to champion.
- Retrained challengers without a promotion did not replace the active champion.
- Bias values near zero indicate more balanced over- and
  under-forecasting.
"""
        )

    st.subheader("Segment performance")

    segment_metric = st.selectbox(
        "Segment metric",
        options=[
            "rmse",
            "mae",
            "wmape_percent",
            "bias",
        ],
        format_func={
            "rmse": "RMSE",
            "mae": "MAE",
            "wmape_percent": "WMAPE (%)",
            "bias": "Bias",
        }.get,
    )

    segment_figure = build_segment_chart(
        segment_metrics,
        metric=segment_metric,
    )
    st.plotly_chart(
        segment_figure,
        use_container_width=True,
    )

    with st.expander(
        "Segment metrics",
        expanded=False,
    ):
        st.dataframe(
            segment_metrics,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "Lifecycle result data",
        expanded=False,
    ):
        tab_without, tab_with = st.tabs(
            [
                "Without retraining",
                "With retraining",
            ]
        )

        with tab_without:
            st.dataframe(
                without_retraining,
                use_container_width=True,
                hide_index=True,
            )

        with tab_with:
            st.dataframe(
                with_retraining,
                use_container_width=True,
                hide_index=True,
            )

    st.caption(f"Results loaded from: {results_path}")


main()
