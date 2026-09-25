import json
import math
from datetime import UTC, datetime

import pandas as pd
from mlflow import MlflowClient

from mlops_sales_forecasting.api.schema import (
    PredictionRequest,
)
from mlops_sales_forecasting.inference.model_loader import (
    load_xgboost_model,
)
from mlops_sales_forecasting.inference.prediction_service import (
    predict_with_bundle,
)
from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_sales_forecasting.inference.serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)
from mlops_sales_forecasting.pipeline.project_factory import (
    build_project_training_pipeline,
)
from mlops_sales_forecasting.pipeline.status import (
    PipelineRunStatus,
)
from mlops_sales_forecasting.tracking.mlflow import (
    start_training_run,
)
from mlops_sales_forecasting.tracking.model_artifact import (
    log_model_artifact,
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


def test_real_rossmann_model_serves_predictions(
    tmp_path,
) -> None:
    config = build_config(tmp_path)
    raw_path = tmp_path / "data" / "raw"
    write_rossmann_sources(raw_path)

    tracking_uri = config["tracking"][
        "mlflow_tracking_uri"
    ]
    experiment_name = config["tracking"][
        "experiment_name"
    ]
    artifact_root = (
        tmp_path / "mlflow-artifacts"
    )
    artifact_root.mkdir()

    client = MlflowClient(
        tracking_uri=tracking_uri,
    )
    client.create_experiment(
        name=experiment_name,
        artifact_location=artifact_root.as_uri(),
    )

    pipeline = build_project_training_pipeline(
        config
    )

    with start_training_run(
        config,
        run_name="rossmann-serving-integration",
    ) as mlflow_run_id:
        result = pipeline.run(
            run_id=mlflow_run_id
        )

        logged_model = log_model_artifact(
            logger=pipeline.model_logger,
            training_result=(
                result.pipeline.training
            ),
            config=config,
            artifact_path="model",
        )

    assert logged_model.model_uri.startswith(
        "models:/m-"
    )

    model = load_xgboost_model(
        logged_model.model_uri
    )
    input_columns = list(
        model.input_columns
    )

    assert input_columns
    assert "Store" in input_columns
    assert "sales_lag_1" in input_columns
    assert "sales_rolling_mean_7" in input_columns

    store_metadata = pd.read_parquet(
        tmp_path
        / "data"
        / "validation"
        / "store.parquet"
    )
    known_calendar = pd.read_parquet(
        tmp_path
        / "data"
        / "features"
        / "known_calendar.parquet"
    )
    store_1_date = (
        known_calendar.loc[
            known_calendar["Store"].eq(1),
            "Date",
        ]
        .max()
    )
    store_2_date = (
        known_calendar.loc[
            known_calendar["Store"].eq(2),
            "Date",
        ]
        .max()
    )

    assert pd.notna(store_1_date)
    assert pd.notna(store_2_date)

    store_state = json.loads(
        (
            tmp_path
            / "artifacts"
            / "models"
            / "latest_state.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    manifest = ServingReleaseManifest(
        schema_version=1,
        release_id="release-integration",
        created_at_utc=datetime.now(
            UTC
        ).isoformat(),
        task_type=TaskType.FORECASTING,
        model=ModelReference(
            name="rossmann-integration-model",
            version="1",
            run_id=mlflow_run_id,
            uri=logged_model.model_uri,
            model_type="xgboost",
        ),
        artifacts={
            "store_metadata": ArtifactReference(
                path="store_metadata.parquet",
                sha256="a" * 64,
            ),
            "store_state": ArtifactReference(
                path="store_state.json",
                sha256="b" * 64,
            ),
            "known_calendar": ArtifactReference(
                path="known_calendar.parquet",
                sha256="c" * 64,
            ),
        },
        metadata={
            "target_transformation": "log1p",
        },
    )

    bundle = ServingBundle(
        release_id="release-integration",
        manifest=manifest,
        model=model,
        serving_alias="champion",
        target_transformation="log1p",
        store_metadata=store_metadata,
        store_state=store_state,
        known_calendar=known_calendar,
    )
    validate_serving_bundle(bundle)

    request = PredictionRequest(
        inputs=[
            {
                "Store": 1,
                "Date": store_1_date.strftime(
                    "%Y-%m-%d"
                ),
                "Open": 1,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            },
            {
                "Store": 2,
                "Date": store_2_date.strftime(
                    "%Y-%m-%d"
                ),
                "Open": 0,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            },
        ]
    )

    response = predict_with_bundle(
        request,
        bundle,
        config,
    )

    assert response.release_id == (
        "release-integration"
    )
    assert len(response.predictions) == 2
    assert math.isfinite(
        response.predictions[0].prediction
    )
    assert (
        response.predictions[0].prediction
        >= 0.0
    )
    assert (
        response.predictions[1].prediction
        == 0.0
    )