"""Shared colors and chart styling for forecasting dashboards."""

import plotly.graph_objects as go

DARK_GREEN = "#112F26"
GREEN = "#008569"
MINT = "#65B8A0"
LIGHT_GREEN = "#DFEFE8"
MUTED_GREEN = "#C5CECA"
BACKGROUND = "#F2F6F4"
WHITE = "#FFFFFF"
TEXT = DARK_GREEN
TERRACOTTA = "#BF735F"

CHART_COLORS = (
    GREEN,
    MINT,
    DARK_GREEN,
    TERRACOTTA,
    MUTED_GREEN,
)

PERFORMANCE_COLORS = {
    "rmse": GREEN,
    "mae": MINT,
    "bias": DARK_GREEN,
}

LIFECYCLE_COLORS = {
    "with_retraining": GREEN,
    "without_retraining": TERRACOTTA,
    "retraining_event": DARK_GREEN,
    "champion_promoted": MINT,
}

CHART_FONT_SIZE = 20
CHART_AXIS_FONT_SIZE = 20
CHART_TICK_FONT_SIZE = 18
CHART_LEGEND_FONT_SIZE = 20
CHART_ANNOTATION_FONT_SIZE = 17
CHART_VALUE_FONT_SIZE = 22


def apply_chart_theme(
    figure: go.Figure,
) -> go.Figure:
    """Apply the shared dashboard theme to a Plotly figure."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        colorway=list(CHART_COLORS),
        font={
            "family": "Arial, sans-serif",
            "color": TEXT,
            "size": CHART_FONT_SIZE,
        },
        legend={
            "font": {
                "size": CHART_LEGEND_FONT_SIZE,
                "color": TEXT,
            },
        },
        hoverlabel={
            "bgcolor": WHITE,
            "bordercolor": MUTED_GREEN,
            "font": {
                "color": TEXT,
                "size": CHART_FONT_SIZE,
            },
        },
    )

    axis_style = {
        "gridcolor": LIGHT_GREEN,
        "linecolor": MUTED_GREEN,
        "zerolinecolor": MUTED_GREEN,
        "title_font": {
            "size": CHART_AXIS_FONT_SIZE,
            "color": TEXT,
        },
        "tickfont": {
            "size": CHART_TICK_FONT_SIZE,
            "color": TEXT,
        },
        "automargin": True,
    }

    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)

    figure.update_annotations(
        font={
            "size": CHART_ANNOTATION_FONT_SIZE,
            "color": TEXT,
        },
    )

    return figure
