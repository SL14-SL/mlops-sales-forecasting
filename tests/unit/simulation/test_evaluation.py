from pathlib import Path

import pandas as pd
import pytest

from mlops_sales_forecasting.simulation.evaluation import (
    build_final_evaluation_frame,
    build_segment_metrics,
    export_runtime_evaluation,
)


def build_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [
                1,
                1,
                2,
                1,
                2,
            ],
            "Date": [
                "2026-09-30",
                "2026-09-30",
                "2026-09-30",
                "2026-10-01",
                "2026-10-01",
            ],
            "prediction": [
                90.0,
                95.0,
                180.0,
                110.0,
                210.0,
            ],
            "release_id": [
                "release-1",
                "release-1",
                "release-1",
                "release-2",
                "release-2",
            ],
            "request_id": [
                "old-request",
                "simulation-day-0001",
                "simulation-day-0001",
                "simulation-day-0002",
                "simulation-day-0002",
            ],
            "row_index": [
                0,
                0,
                1,
                0,
                1,
            ],
            "timestamp": [
                "2026-09-30T09:00:00Z",
                "2026-09-30T10:00:00Z",
                "2026-09-30T10:00:00Z",
                "2026-10-01T10:00:00Z",
                "2026-10-01T10:00:00Z",
            ],
        }
    )


def build_ground_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [
                1,
                2,
                1,
                2,
            ],
            "Date": [
                "2026-09-30",
                "2026-09-30",
                "2026-10-01",
                "2026-10-01",
            ],
            "Sales": [
                100.0,
                200.0,
                120.0,
                220.0,
            ],
            "Open": [
                1,
                1,
                1,
                0,
            ],
            "Promo": [
                1,
                0,
                1,
                0,
            ],
        }
    )


def test_builds_final_open_store_evaluation() -> None:
    result = build_final_evaluation_frame(
        predictions=build_predictions(),
        ground_truth=build_ground_truth(),
    )

    assert len(result) == 3
    assert result["prediction"].tolist() == [
        95.0,
        180.0,
        110.0,
    ]
    assert result["Open"].eq(1).all()


def test_builds_segment_metrics() -> None:
    evaluation = build_final_evaluation_frame(
        predictions=build_predictions(),
        ground_truth=build_ground_truth(),
    )

    result = build_segment_metrics(
        evaluation,
        model_variant="Managed lifecycle",
    )

    assert result["segment"].tolist() == [
        "All open stores",
        "Promo stores",
        "Non-promo stores",
    ]
    assert result["rows"].tolist() == [
        3,
        2,
        1,
    ]

    all_stores = result.iloc[0]

    assert all_stores["rmse"] == pytest.approx(13.2287565553)
    assert all_stores["mae"] == pytest.approx(11.6666666667)
    assert all_stores["bias"] == pytest.approx(11.6666666667)
    assert all_stores["wmape_percent"] == pytest.approx(8.3333333333)


def test_exports_runtime_evaluation(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    predictions_path = runtime / "predictions" / "history" / "date=2026-10-01"
    monitoring_path = runtime / "monitoring"

    predictions_path.mkdir(
        parents=True,
    )
    monitoring_path.mkdir(
        parents=True,
    )

    build_predictions().to_parquet(
        predictions_path / "part.parquet",
        index=False,
    )
    build_ground_truth().to_csv(
        monitoring_path / "cumulative_ground_truth.csv",
        index=False,
    )

    output = tmp_path / "evaluation.parquet"

    result = export_runtime_evaluation(
        runtime_root=runtime,
        output_path=output,
    )

    assert output.is_file()
    assert len(result) == 3


def test_rejects_invalid_window() -> None:
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        build_final_evaluation_frame(
            predictions=build_predictions(),
            ground_truth=build_ground_truth(),
            window_days=0,
        )
