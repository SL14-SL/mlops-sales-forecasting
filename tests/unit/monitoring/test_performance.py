from pathlib import Path

import pandas as pd
import pytest

from mlops_sales_forecasting.monitoring.performance import (
    compute_regression_metrics,
    compute_rolling_metrics,
    evaluate_predictions,
    load_table,
    prepare_evaluation_frame,
    save_metrics,
)


def test_compute_regression_metrics() -> None:
    frame = pd.DataFrame(
        {
            "Sales": [
                100.0,
                200.0,
            ],
            "prediction": [
                90.0,
                220.0,
            ],
        }
    )

    result = compute_regression_metrics(frame)

    assert result["rmse"] == pytest.approx(15.8113883008)
    assert result["mae"] == pytest.approx(15.0)
    assert result["bias"] == pytest.approx(-5.0)
    assert result["n_samples"] == 2


def test_compute_regression_metrics_drops_missing_rows() -> None:
    frame = pd.DataFrame(
        {
            "Sales": [
                100.0,
                None,
            ],
            "prediction": [
                90.0,
                200.0,
            ],
        }
    )

    result = compute_regression_metrics(frame)

    assert result["n_samples"] == 1
    assert result["rmse"] == pytest.approx(10.0)


def test_compute_regression_metrics_rejects_missing_column() -> None:
    with pytest.raises(
        KeyError,
        match="missing columns",
    ):
        compute_regression_metrics(
            pd.DataFrame(
                {
                    "Sales": [
                        100.0,
                    ],
                }
            )
        )


def test_compute_rolling_metrics() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-09-01",
                    "2026-09-02",
                    "2026-09-03",
                ]
            ),
            "Sales": [
                100.0,
                110.0,
                120.0,
            ],
            "prediction": [
                90.0,
                100.0,
                110.0,
            ],
        }
    )

    result = compute_rolling_metrics(
        frame,
        window="2D",
    )

    assert len(result) == 3
    assert result.iloc[-1]["n_samples"] == 2
    assert result.iloc[-1]["rmse"] == pytest.approx(10.0)
    assert result.iloc[-1]["window_end"] == pd.Timestamp("2026-09-03")


def test_compute_rolling_metrics_applies_minimum_samples() -> None:
    frame = pd.DataFrame(
        {
            "Date": [
                "2026-09-01",
                "2026-09-02",
            ],
            "Sales": [
                100.0,
                110.0,
            ],
            "prediction": [
                90.0,
                100.0,
            ],
        }
    )

    result = compute_rolling_metrics(
        frame,
        window="7D",
        minimum_samples=2,
    )

    assert len(result) == 1
    assert result.iloc[0]["n_samples"] == 2


def test_prepare_evaluation_frame() -> None:
    predictions = pd.DataFrame(
        {
            "Store": [
                "1",
                "2",
            ],
            "Date": [
                "2026-09-01",
                "2026-09-01",
            ],
            "prediction": [
                95.0,
                205.0,
            ],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "Date": pd.to_datetime(
                [
                    "2026-09-01",
                    "2026-09-01",
                ]
            ),
            "Sales": [
                100.0,
                200.0,
            ],
        }
    )

    result = prepare_evaluation_frame(
        predictions=predictions,
        ground_truth=ground_truth,
    )

    assert len(result) == 2
    assert result["Store"].tolist() == [
        1,
        2,
    ]
    assert result["Sales"].tolist() == [
        100.0,
        200.0,
    ]


def test_prepare_evaluation_frame_rejects_duplicates() -> None:
    predictions = pd.DataFrame(
        {
            "Store": [
                1,
                1,
            ],
            "Date": [
                "2026-09-01",
                "2026-09-01",
            ],
            "prediction": [
                90.0,
                95.0,
            ],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "Store": [
                1,
            ],
            "Date": [
                "2026-09-01",
            ],
            "Sales": [
                100.0,
            ],
        }
    )

    with pytest.raises(
        pd.errors.MergeError,
        match="one-to-one",
    ):
        prepare_evaluation_frame(
            predictions=predictions,
            ground_truth=ground_truth,
        )


@pytest.mark.parametrize(
    "suffix",
    [
        ".csv",
        ".parquet",
    ],
)
def test_save_and_load_metrics(
    tmp_path: Path,
    suffix: str,
) -> None:
    metrics = pd.DataFrame(
        {
            "rmse": [
                10.0,
            ],
            "mae": [
                8.0,
            ],
            "bias": [
                -2.0,
            ],
            "n_samples": [
                5,
            ],
        }
    )
    output_path = tmp_path / "nested" / f"performance{suffix}"

    save_metrics(
        metrics,
        output_path,
    )
    loaded = load_table(output_path)

    pd.testing.assert_frame_equal(
        loaded,
        metrics,
    )


def test_evaluate_predictions_writes_rolling_metrics(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    ground_truth_path = tmp_path / "ground_truth.csv"
    output_path = tmp_path / "monitoring" / "performance_rolling.parquet"

    pd.DataFrame(
        {
            "Store": [
                1,
                1,
            ],
            "Date": [
                "2026-09-01",
                "2026-09-02",
            ],
            "prediction": [
                90.0,
                110.0,
            ],
        }
    ).to_parquet(
        predictions_path,
        index=False,
    )
    pd.DataFrame(
        {
            "Store": [
                1,
                1,
            ],
            "Date": [
                "2026-09-01",
                "2026-09-02",
            ],
            "Sales": [
                100.0,
                120.0,
            ],
        }
    ).to_csv(
        ground_truth_path,
        index=False,
    )

    result = evaluate_predictions(
        predictions_path=predictions_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        rolling_window="7D",
    )

    assert output_path.is_file()
    assert len(result) == 2
    assert result.iloc[-1]["n_samples"] == 2
    assert result.iloc[-1]["rmse"] == pytest.approx(10.0)
