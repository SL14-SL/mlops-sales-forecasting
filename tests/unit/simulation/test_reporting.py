import pandas as pd
import pytest

from mlops_sales_forecasting.simulation.contracts import (
    SimulationDayResult,
)
from mlops_sales_forecasting.simulation.reporting import (
    load_lifecycle_results,
    results_to_frame,
    summarize_simulation_comparison,
    write_lifecycle_results,
)


def build_result(
    *,
    day: int,
    rmse: float,
    retraining_enabled: bool,
    event: str | None = None,
    champion_promoted: bool = False,
) -> SimulationDayResult:
    return SimulationDayResult(
        day=day,
        cumulative_days=day,
        rmse=rmse,
        mae=rmse * 0.75,
        bias=-10.0,
        n_samples=100,
        window_start="2026-01-01",
        window_end="2026-01-07",
        event=event,
        champion_promoted=(champion_promoted),
        scenario="gradual_promo_shift",
        retraining_enabled=(retraining_enabled),
        drift_start_day=20,
        drift_duration_days=14,
        maximum_base_uplift=0.0,
        maximum_promo_uplift=-0.25,
    )


def test_results_to_frame() -> None:
    frame = results_to_frame(
        [
            build_result(
                day=1,
                rmse=1000.0,
                retraining_enabled=True,
            ),
        ]
    )

    assert len(frame) == 1
    assert frame.iloc[0]["rmse"] == 1000.0
    assert bool(frame.iloc[0]["retraining_enabled"])


def test_write_and_load_lifecycle_results(
    tmp_path,
) -> None:
    path = tmp_path / "lifecycle.csv"
    expected = [
        build_result(
            day=1,
            rmse=1000.0,
            retraining_enabled=True,
        ),
        build_result(
            day=2,
            rmse=800.0,
            retraining_enabled=True,
            event="retrain",
            champion_promoted=True,
        ),
    ]

    write_lifecycle_results(
        expected,
        path,
    )
    loaded = load_lifecycle_results(path)

    assert loaded["day"].tolist() == [
        1,
        2,
    ]
    assert loaded["event"].tolist() == [
        None,
        "retrain",
    ]
    assert loaded["champion_promoted"].tolist() == [
        False,
        True,
    ]


def test_summarize_simulation_comparison() -> None:
    without = results_to_frame(
        [
            build_result(
                day=1,
                rmse=1000.0,
                retraining_enabled=False,
            ),
            build_result(
                day=2,
                rmse=2000.0,
                retraining_enabled=False,
                event="would_retrain",
            ),
        ]
    )
    with_run = results_to_frame(
        [
            build_result(
                day=1,
                rmse=1000.0,
                retraining_enabled=True,
            ),
            build_result(
                day=2,
                rmse=1000.0,
                retraining_enabled=True,
                event="retrain",
                champion_promoted=True,
            ),
        ]
    )

    summary = summarize_simulation_comparison(
        without,
        with_run,
    )

    assert summary["final_rmse_without_retraining"] == 2000.0
    assert summary["final_rmse_with_retraining"] == 1000.0
    assert summary["relative_rmse_improvement"] == 0.5
    assert summary["retraining_events"] == 1
    assert summary["promotion_events"] == 1


def test_invalid_metric_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        build_result(
            day=1,
            rmse=float("nan"),
            retraining_enabled=True,
        )


def test_missing_result_file_is_rejected(
    tmp_path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        load_lifecycle_results(tmp_path / "missing.csv")


def test_empty_comparison_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must contain results",
    ):
        summarize_simulation_comparison(
            pd.DataFrame(),
            pd.DataFrame(),
        )
