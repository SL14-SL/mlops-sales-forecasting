import json

import pandas as pd

from mlops_sales_forecasting.pipeline.project_factory import (
    build_project_training_pipeline,
)
from mlops_sales_forecasting.pipeline.status import (
    PipelineRunStatus,
)
from mlops_sales_forecasting.tracking.mlflow import (
    start_training_run,
)


def write_rossmann_sources(
    raw_path,
) -> None:
    raw_path.mkdir(parents=True)

    rows = []

    for date in pd.date_range(
        "2026-01-01",
        periods=40,
        freq="D",
    ):
        for store in (
            1,
            2,
        ):
            promo = int(date.day % 2 == 0)
            rows.append(
                {
                    "Store": store,
                    "Date": date,
                    "Sales": float(1000 + store * 100 + date.day * 10 + promo * 50),
                    "Customers": (100 + store * 10 + date.day),
                    "Open": 1,
                    "Promo": promo,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                    "DayOfWeek": (date.dayofweek + 1),
                }
            )

    pd.DataFrame(rows).to_csv(
        raw_path / "train.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "StoreType": [
                "a",
                "b",
            ],
            "Assortment": [
                "a",
                "c",
            ],
            "CompetitionDistance": [
                500.0,
                750.0,
            ],
            "CompetitionOpenSinceMonth": [
                1,
                1,
            ],
            "CompetitionOpenSinceYear": [
                2025,
                2025,
            ],
            "Promo2SinceWeek": [
                1,
                1,
            ],
            "Promo2SinceYear": [
                2025,
                2025,
            ],
            "PromoInterval": [
                "Jan,Apr,Jul,Oct",
                "Feb,May,Aug,Nov",
            ],
        }
    ).to_csv(
        raw_path / "store.csv",
        index=False,
    )


def build_config(
    tmp_path,
) -> dict:
    return {
        "environment": "test",
        "random_seed": 42,
        "paths": {
            "raw_data": str(tmp_path / "data" / "raw"),
            "validated_data": str(tmp_path / "data" / "validation"),
            "features": str(tmp_path / "data" / "features"),
            "splits": str(tmp_path / "data" / "splits"),
            "models": str(tmp_path / "artifacts" / "models"),
            "artifacts": str(tmp_path / "artifacts"),
        },
        "tracking": {
            "mlflow_tracking_uri": (
                "sqlite:///"
                f"{tmp_path / 'mlflow.db'}"
            ),
            "experiment_name": ("rossmann-integration-test"),
        },
        "data": {
            "target_column": "Sales",
            "known_targets": [
                "Sales",
                "Customers",
            ],
            "time_column": "Date",
            "id_columns": [
                "Store",
            ],
            "primary_keys": [
                "Store",
                "Date",
            ],
        },
        "features": {
            "enabled_steps": [
                "sort",
                "temporal",
                "lags",
                "competition",
                "promo",
                "cast_categoricals",
                "drop_technical",
                "drop_configured",
            ],
            "lag_features": {
                "lags": [
                    1,
                    7,
                ],
                "rolling_windows": [
                    7,
                ],
            },
            "technical_drop_columns": [
                "CompetitionOpenSinceMonth",
                "CompetitionOpenSinceYear",
                "Promo2SinceWeek",
                "Promo2SinceYear",
                "PromoInterval",
            ],
            "drop_columns": [],
        },
        "training": {
            "normal_validation_days": 5,
            "drift_validation_days": 3,
            "target_transformation": "log1p",
            "is_drift_run": False,
            "recency_weighting": {
                "enabled_for_drift": True,
                "promo_column": "Promo",
                "last_30_days_weight": 5.0,
                "last_60_days_weight": 3.0,
                "last_120_days_weight": 1.5,
                "default_weight": 1.0,
            },
        },
        "metrics": {
            "primary": "rmse",
            "evaluate_on_original_scale": True,
        },
        "promotion": {
            "minimum_validation_rows": 1,
            "minimum_segment_rows": 1,
            "required_segments": [
                "promo",
                "non_promo",
            ],
        },
        "model": {
            "type": "xgboost",
            "registry_name": ("rossmann-integration-model"),
            "params": {
                "tree_method": "hist",
                "enable_categorical": True,
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "objective": ("reg:squarederror"),
                "early_stopping_rounds": 3,
            },
        },
    }


def test_real_rossmann_training_pipeline(
    tmp_path,
) -> None:
    config = build_config(tmp_path)
    raw_path = tmp_path / "data" / "raw"
    write_rossmann_sources(raw_path)

    pipeline = build_project_training_pipeline(config)

    with start_training_run(
        config,
        run_name="rossmann-integration",
    ) as mlflow_run_id:
        result = pipeline.run(run_id=mlflow_run_id)

    assert result.run.status is PipelineRunStatus.SUCCEEDED
    assert result.run.run_id == mlflow_run_id
    assert result.pipeline.training.run_id == mlflow_run_id
    assert result.pipeline.evaluation.approved is True
    assert result.pipeline.evaluation.metrics["rmse"] >= 0

    assert (tmp_path / "data" / "validation" / "train.parquet").is_file()
    assert (tmp_path / "data" / "validation" / "store.parquet").is_file()
    assert (tmp_path / "data" / "features" / "features.parquet").is_file()
    assert (tmp_path / "data" / "features" / "known_calendar.parquet").is_file()

    state_path = tmp_path / "artifacts" / "models" / "latest_state.json"
    assert state_path.is_file()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state) == {
        "1",
        "2",
    }

    pipeline_state_path = tmp_path / "artifacts" / "pipeline-runs" / f"{mlflow_run_id}.json"
    assert pipeline_state_path.is_file()
