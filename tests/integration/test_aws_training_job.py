import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error

from mlops_sales_forecasting.aws.train import run_training_job
from mlops_sales_forecasting.configs.paths import get_project_root
from mlops_sales_forecasting.data.splits.split import RossmannDatasetSplitter
from mlops_sales_forecasting.training.dataset import normalize_feature_dtypes
from mlops_sales_forecasting.training.evaluate_metrics import align_features_for_evaluation
from mlops_sales_forecasting.training.target_transform import inverse_transform_target
from mlops_sales_forecasting.training.utils import build_drop_columns


@pytest.fixture
def job_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load((get_project_root() / "configs/dev.yaml").read_text(encoding="utf-8"))
    config["random_seed"] = 42
    config["model"]["params"].update(
        n_estimators=5,
        max_depth=2,
        n_jobs=1,
        early_stopping_rounds=2,
    )
    config["training"]["normal_validation_days"] = 5
    config["promotion"]["minimum_validation_rows"] = 1
    config["promotion"]["minimum_segment_rows"] = 1

    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    input_directory = tmp_path / "input"
    input_directory.mkdir()

    rows = []
    for day_index, day in enumerate(pd.date_range("2026-01-01", periods=80)):
        for store in (1, 2):
            promo = day_index % 2
            rows.append(
                {
                    "Store": store,
                    "Date": day,
                    "Sales": float(1000 + store * 100 + day_index * 5 + promo * 50),
                    "Customers": 100 + store * 10,
                    "Open": 1,
                    "Promo": promo,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                    "DayOfWeek": day.dayofweek + 1,
                }
            )

    pd.DataFrame(rows).to_csv(input_directory / "train.csv", index=False)
    pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "c"],
            "CompetitionDistance": [500.0, 750.0],
            "CompetitionOpenSinceMonth": [1, 1],
            "CompetitionOpenSinceYear": [2025, 2025],
            "Promo2SinceWeek": [1, 1],
            "Promo2SinceYear": [2025, 2025],
            "PromoInterval": ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov"],
        }
    ).to_csv(input_directory / "store.csv", index=False)

    return config_path, input_directory


def test_training_job_exports_reloadable_candidate(
    tmp_path: Path,
    job_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, input_directory = job_inputs
    before = {path.name: path.read_bytes() for path in input_directory.iterdir()}
    work_directory = tmp_path / "work"
    model_directory = tmp_path / "model"
    output_directory = tmp_path / "output"
    original_tracking_uri = mlflow.get_tracking_uri()

    # Keep any library-created relative files inside the test workspace.
    monkeypatch.chdir(tmp_path)

    try:
        report = run_training_job(
            config_path=config_path,
            input_directory=input_directory,
            work_directory=work_directory,
            model_directory=model_directory,
            output_directory=output_directory,
            estimators=3,
        )

        persisted_report = json.loads(
            (output_directory / "evaluation.json").read_text(encoding="utf-8")
        )
        resolved_config = json.loads(
            (output_directory / "resolved_config.json").read_text(encoding="utf-8")
        )

        assert persisted_report == report
        assert report["evaluation_approved"] is True
        assert report["promoted"] is False
        assert report["parameters"]["model_n_estimators"] == 3
        assert report["training_rows"] > 0
        assert report["validation_rows"] > 0

        assert (model_directory / "model/MLmodel").is_file()
        assert (model_directory / "model/model.ubj").is_file()

        state = json.loads((model_directory / "latest_state.json").read_text(encoding="utf-8"))
        assert state["schema_version"] == 1
        assert set(state["stores"]) == {"1", "2"}
        assert not pd.read_parquet(model_directory / "known_calendar.parquet").empty

        features = pd.read_parquet(work_directory / "data/features/features.parquet")
        validation = RossmannDatasetSplitter().split(features, resolved_config).validation
        assert len(validation) == report["validation_rows"]

        model = mlflow.xgboost.load_model(str(model_directory / "model"))
        model_input = normalize_feature_dtypes(
            validation.drop(
                columns=build_drop_columns(resolved_config),
                errors="ignore",
            )
        )
        model_input = align_features_for_evaluation(model, model_input)
        predictions = inverse_transform_target(
            model.predict(model_input),
            resolved_config["training"]["target_transformation"],
        )

        assert np.isfinite(predictions).all()
        rmse = float(np.sqrt(mean_squared_error(validation["Sales"], predictions)))
        assert rmse == pytest.approx(
            report["evaluation_metrics"]["rmse"],
            rel=1e-8,
            abs=1e-6,
        )

        client = MlflowClient(tracking_uri=resolved_config["tracking"]["mlflow_tracking_uri"])
        assert list(client.search_registered_models()) == []
        assert not (model_directory / "active_serving_release.json").exists()
        assert not (work_directory / "models/active_serving_release.json").exists()

        after = {path.name: path.read_bytes() for path in input_directory.iterdir()}
        assert after == before
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)


@pytest.mark.parametrize("occupied_directory", ["work", "model", "output"])
def test_training_job_preserves_existing_results(
    tmp_path: Path,
    job_inputs: tuple[Path, Path],
    occupied_directory: str,
) -> None:
    config_path, input_directory = job_inputs
    occupied = tmp_path / occupied_directory
    occupied.mkdir()
    marker = occupied / "existing-result.txt"
    marker.write_text("keep this result", encoding="utf-8")

    with pytest.raises(ValueError, match="Job directory must be empty"):
        run_training_job(
            config_path=config_path,
            input_directory=input_directory,
            work_directory=tmp_path / "work",
            model_directory=tmp_path / "model",
            output_directory=tmp_path / "output",
        )

    assert marker.read_text(encoding="utf-8") == "keep this result"
    assert not (tmp_path / "output/evaluation.json").exists()
