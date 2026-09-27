import pandas as pd
import pytest

from mlops_sales_forecasting.monitoring.dashboard import (
    build_drift_chart,
    build_monitoring_paths,
    build_performance_chart,
)


def test_build_monitoring_paths() -> None:
    result = build_monitoring_paths(
        {
            "paths": {
                "monitoring": ("data/monitoring"),
            },
        }
    )

    assert result == {
        "performance": ("data/monitoring/performance_rolling.parquet"),
        "feature_drift": ("data/monitoring/feature_drift_history.parquet"),
    }


def test_build_performance_chart() -> None:
    history = pd.DataFrame(
        {
            "window_end": [
                "2026-09-25T00:00:00Z",
                "2026-09-26T00:00:00Z",
            ],
            "rmse": [
                800.0,
                750.0,
            ],
            "mae": [
                600.0,
                550.0,
            ],
            "bias": [
                20.0,
                -10.0,
            ],
        }
    )

    figure = build_performance_chart(history)

    assert len(figure.data) == 3
    assert [trace.name for trace in figure.data] == [
        "RMSE",
        "MAE",
        "Bias",
    ]


def test_build_drift_chart_uses_latest_window() -> None:
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-09-25T00:00:00Z",
                "2026-09-26T00:00:00Z",
                "2026-09-26T00:00:00Z",
            ],
            "feature": [
                "Promo",
                "Promo",
                "StoreType",
            ],
            "score": [
                0.1,
                0.3,
                0.2,
            ],
            "drift_detected": [
                False,
                True,
                False,
            ],
        }
    )

    figure = build_drift_chart(history)

    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [
        "Promo",
        "StoreType",
    ]
    assert list(figure.data[0].y) == [
        0.3,
        0.2,
    ]


@pytest.mark.parametrize(
    ("builder", "history"),
    [
        (
            build_performance_chart,
            pd.DataFrame(
                {
                    "rmse": [
                        1.0,
                    ],
                }
            ),
        ),
        (
            build_drift_chart,
            pd.DataFrame(
                {
                    "feature": [
                        "Promo",
                    ],
                }
            ),
        ),
    ],
)
def test_chart_rejects_invalid_history(
    builder,
    history: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="required columns",
    ):
        builder(history)
