from pathlib import Path

import pandas as pd

from mlops_sales_forecasting.monitoring.monitoring_refresh import (
    load_inference_history,
    rebuild_cumulative_ground_truth,
    refresh_monitoring_signals,
)


def test_rebuilds_cumulative_ground_truth(
    tmp_path: Path,
) -> None:
    first_batch = tmp_path / "ground_truth_001.csv"
    second_batch = tmp_path / "ground_truth_002.csv"
    output_path = tmp_path / "monitoring" / "cumulative_ground_truth.csv"

    pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "Date": [
                "2026-09-01",
                "2026-09-01",
            ],
            "Sales": [
                100.0,
                200.0,
            ],
        }
    ).to_csv(
        first_batch,
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
                110.0,
                120.0,
            ],
        }
    ).to_csv(
        second_batch,
        index=False,
    )

    result = rebuild_cumulative_ground_truth(
        [
            str(first_batch),
            str(second_batch),
        ],
        output_path=str(output_path),
    )

    assert len(result) == 3
    assert output_path.is_file()

    updated_value = result.loc[
        (result["Store"] == 1) & (result["Date"] == pd.Timestamp("2026-09-01")),
        "Sales",
    ].iloc[0]

    assert updated_value == 110.0


def test_load_inference_history_deduplicates_request_rows(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"

    common = {
        "Store": [
            1,
        ],
        "Date": [
            "2026-09-01",
        ],
        "request_id": [
            "request-1",
        ],
        "row_index": [
            0,
        ],
    }

    pd.DataFrame(
        {
            **common,
            "prediction": [
                90.0,
            ],
            "timestamp": [
                "2026-09-01T10:00:00Z",
            ],
        }
    ).to_parquet(
        first_path,
        index=False,
    )
    pd.DataFrame(
        {
            **common,
            "prediction": [
                95.0,
            ],
            "timestamp": [
                "2026-09-01T11:00:00Z",
            ],
        }
    ).to_parquet(
        second_path,
        index=False,
    )

    result = load_inference_history(
        [
            str(first_path),
            str(second_path),
        ]
    )

    assert len(result) == 1
    assert result.iloc[0]["prediction"] == 95.0


def test_refresh_without_operational_data_is_safe(
    tmp_path: Path,
) -> None:
    config = {
        "paths": {
            "raw_data": str(tmp_path / "raw"),
            "predictions": str(tmp_path / "predictions"),
            "monitoring": str(tmp_path / "monitoring"),
            "features": str(tmp_path / "features"),
        },
        "monitoring": {
            "feature_drift": {
                "numeric_features": [
                    "CompetitionDistance",
                ],
                "categorical_features": [
                    "Promo",
                ],
                "minimum_samples": 50,
                "p_value_threshold": 0.01,
                "statistic_threshold": 0.10,
                "lookback_days": 14,
            },
            "retraining": {
                "minimum_new_training_rows": 500,
                "performance": {
                    "rolling_window": "7D",
                    "minimum_samples": 500,
                    "open_store_only": True,
                },
            },
        },
    }

    result = refresh_monitoring_signals(config=config)

    assert result.ground_truth_rows == 0
    assert result.inference_rows == 0
    assert result.performance_updated is False
    assert result.feature_drift_updated is False
    assert result.performance_reason == ("No Ground-Truth batches available.")


def test_refreshes_performance_and_drift(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw"
    predictions_path = tmp_path / "predictions"
    monitoring_path = tmp_path / "monitoring"
    features_path = tmp_path / "features"

    batch_directory = raw_path / "new_batches"
    inference_directory = predictions_path / "history" / "date=2026-09-27"

    batch_directory.mkdir(parents=True)
    inference_directory.mkdir(parents=True)
    features_path.mkdir(parents=True)

    pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "Date": [
                "2026-09-27",
                "2026-09-27",
            ],
            "Sales": [
                100.0,
                200.0,
            ],
            "Open": [
                1,
                0,
            ],
        }
    ).to_csv(
        batch_directory / "ground_truth_001.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "Date": [
                "2026-09-27",
                "2026-09-27",
            ],
            "prediction": [
                90.0,
                190.0,
            ],
            "request_id": [
                "request-1",
                "request-1",
            ],
            "row_index": [
                0,
                1,
            ],
            "timestamp": [
                "2026-09-27T05:00:00Z",
                "2026-09-27T05:00:00Z",
            ],
            "CompetitionDistance": [
                1000.0,
                2000.0,
            ],
            "Promo": [
                0,
                1,
            ],
        }
    ).to_parquet(
        inference_directory / "request-1.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "CompetitionDistance": [
                100.0,
                200.0,
            ],
            "Promo": [
                0,
                0,
            ],
        }
    ).to_parquet(
        features_path / "features.parquet",
        index=False,
    )

    config = {
        "paths": {
            "raw_data": str(raw_path),
            "predictions": str(predictions_path),
            "monitoring": str(monitoring_path),
            "features": str(features_path),
        },
        "monitoring": {
            "feature_drift": {
                "numeric_features": [
                    "CompetitionDistance",
                ],
                "categorical_features": [
                    "Promo",
                ],
                "minimum_samples": 1,
                "p_value_threshold": 0.01,
                "statistic_threshold": 0.10,
                "lookback_days": 14,
            },
            "retraining": {
                "minimum_new_training_rows": 1,
                "performance": {
                    "rolling_window": "7D",
                    "minimum_samples": 1,
                    "open_store_only": True,
                },
            },
        },
    }

    result = refresh_monitoring_signals(config=config)

    assert result.ground_truth_rows == 2
    assert result.inference_rows == 2
    assert result.performance_updated is True
    assert result.performance_rows == 1
    assert result.feature_drift_updated is True
    assert result.feature_drift_rows == 2

    assert (monitoring_path / "performance_rolling.parquet").is_file()
    assert (monitoring_path / "feature_drift_history.parquet").is_file()

    result = refresh_monitoring_signals(config=config)

    assert result.ground_truth_rows == 2
    assert result.inference_rows == 2
    assert result.performance_updated is True
    assert result.performance_rows == 1
    assert result.feature_drift_updated is True
    assert result.feature_drift_rows == 2

    performance_path = monitoring_path / "performance_rolling.parquet"
    drift_path = monitoring_path / "feature_drift_history.parquet"

    assert performance_path.is_file()
    assert drift_path.is_file()

    performance = pd.read_parquet(performance_path)

    latest_performance = performance.iloc[-1]

    assert latest_performance["n_samples"] == 1
    assert latest_performance["rmse"] == 10.0
    assert latest_performance["mae"] == 10.0
    assert latest_performance["bias"] == 10.0
