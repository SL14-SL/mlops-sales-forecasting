from pathlib import Path

import pandas as pd
import pytest

from mlops_sales_forecasting.monitoring.simulation_dashboard import (
    build_lifecycle_metric_chart,
    build_segment_chart,
    build_simulation_summary,
    load_reference_simulation,
    load_simulation_frame,
)


def build_lifecycle_frame(
    *,
    final_rmse: float,
    event: str = "",
    promoted: bool = False,
    retraining_enabled: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": [
                1,
                2,
            ],
            "cumulative_days": [
                1,
                2,
            ],
            "rmse": [
                1000.0,
                final_rmse,
            ],
            "mae": [
                800.0,
                700.0,
            ],
            "bias": [
                100.0,
                20.0,
            ],
            "n_samples": [
                100,
                200,
            ],
            "window_start": [
                "2015-04-28",
                "2015-04-28",
            ],
            "window_end": [
                "2015-04-28",
                "2015-04-29",
            ],
            "event": [
                "",
                event,
            ],
            "champion_promoted": [
                False,
                promoted,
            ],
            "scenario": [
                "gradual_promo_shift",
                "gradual_promo_shift",
            ],
            "retraining_enabled": [
                retraining_enabled,
                retraining_enabled,
            ],
            "drift_start_day": [
                20,
                20,
            ],
            "drift_duration_days": [
                14,
                14,
            ],
            "maximum_base_uplift": [
                0.0,
                0.0,
            ],
            "maximum_promo_uplift": [
                -0.25,
                -0.25,
            ],
        }
    )


def build_segment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_variant": [
                "Without retraining",
                "With retraining",
            ],
            "segment": [
                "Promo stores",
                "Promo stores",
            ],
            "rows": [
                100,
                100,
            ],
            "rmse": [
                2000.0,
                1000.0,
            ],
            "mae": [
                1500.0,
                700.0,
            ],
            "wmape_percent": [
                30.0,
                14.0,
            ],
            "bias": [
                1200.0,
                100.0,
            ],
        }
    )


def test_load_simulation_frame_sorts_days(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.csv"
    frame = build_lifecycle_frame(
        final_rmse=500.0,
    )
    frame.iloc[::-1].to_csv(
        path,
        index=False,
    )

    result = load_simulation_frame(path)

    assert result["day"].tolist() == [
        1,
        2,
    ]


def test_load_reference_simulation(
    tmp_path: Path,
) -> None:
    lifecycle = build_lifecycle_frame(
        final_rmse=500.0,
    )
    segments = build_segment_frame()

    lifecycle.to_csv(
        tmp_path / "without_retraining.csv",
        index=False,
    )
    lifecycle.to_csv(
        tmp_path / "with_retraining.csv",
        index=False,
    )
    segments.to_csv(
        tmp_path / "segment_metrics.csv",
        index=False,
    )

    without, with_run, result_segments = load_reference_simulation(tmp_path)

    assert len(without) == 2
    assert len(with_run) == 2
    assert len(result_segments) == 2


def test_build_simulation_summary() -> None:
    without = build_lifecycle_frame(
        final_rmse=2000.0,
    )
    with_run = build_lifecycle_frame(
        final_rmse=1000.0,
        event="retrain",
        promoted=True,
        retraining_enabled=True,
    )

    result = build_simulation_summary(
        without,
        with_run,
    )

    assert result["final_rmse_without_retraining"] == 2000.0
    assert result["final_rmse_with_retraining"] == 1000.0
    assert result["relative_rmse_improvement"] == pytest.approx(0.5)
    assert result["retraining_events"] == 1
    assert result["promotion_events"] == 1


@pytest.mark.parametrize(
    "metric",
    [
        "rmse",
        "mae",
        "bias",
    ],
)
def test_build_lifecycle_metric_chart(
    metric: str,
) -> None:
    without = build_lifecycle_frame(
        final_rmse=2000.0,
    )
    with_run = build_lifecycle_frame(
        final_rmse=1000.0,
        event="retrain",
        promoted=True,
        retraining_enabled=True,
    )

    figure = build_lifecycle_metric_chart(
        without,
        with_run,
        metric=metric,
    )

    assert len(figure.data) == 4
    assert figure.data[0].name == ("With retraining")
    assert figure.data[1].name == ("Without retraining")
    assert figure.data[1].line.dash == "dash"
    assert figure.data[2].name == ("Retraining event")
    assert figure.data[3].name == ("Challenger promoted")
    assert list(figure.data[3].x) == [
        2,
    ]


def test_build_segment_chart() -> None:
    figure = build_segment_chart(
        build_segment_frame(),
        metric="wmape_percent",
    )

    assert len(figure.data) == 2
    assert figure.layout.yaxis.title.text == ("WMAPE (%)")


def test_invalid_metric_is_rejected() -> None:
    frame = build_lifecycle_frame(
        final_rmse=500.0,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported simulation metric",
    ):
        build_lifecycle_metric_chart(
            frame,
            frame,
            metric="unknown",
        )
