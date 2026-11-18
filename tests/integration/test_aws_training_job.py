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

from mlops_sales_forecasting.aws.serving_release import prepare_local_serving_release
from mlops_sales_forecasting.aws.train import run_training_job
from mlops_sales_forecasting.configs.paths import get_project_root
from mlops_sales_forecasting.data.splits.split import RossmannDatasetSplitter
from mlops_sales_forecasting.inference.releases.storage import resolve_artifact_uri
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
        store_metadata = pd.read_parquet(model_directory / "store_metadata.parquet")
        assert set(store_metadata["Store"]) == {1, 2}
        assert {
            "Store",
            "StoreType",
            "Assortment",
            "CompetitionDistance",
        }.issubset(store_metadata.columns)

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

        serving_directory = tmp_path.parent / f"{tmp_path.name}-serving"

        bundle = prepare_local_serving_release(
            candidate_directory=tmp_path,
            models_directory=serving_directory,
            model_name="integration-candidate",
        )

        assert bundle.model_name == "integration-candidate"
        assert bundle.model_run_id == report["run_id"]
        assert bundle.target_transformation == "log1p"
        assert set(bundle.store_metadata["Store"]) == {1, 2}
        assert bundle.manifest.metadata["registry_registered"] is False
        assert bundle.manifest.metadata["purpose"] == "serving_smoke_test"
        assert not (serving_directory / "active_serving_release.json").exists()

        release_root = serving_directory / "serving_releases" / bundle.release_id

        for reference in bundle.manifest.artifacts.values():
            resolve_artifact_uri(
                release_root=str(release_root),
                reference=reference,
            )

        reloaded_predictions = bundle.model.predict(model_input)
        np.testing.assert_allclose(
            reloaded_predictions,
            model.predict(model_input),
        )
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


def test_shared_lifecycle_portable_release_and_registry_rollback(
    tmp_path: Path,
    job_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real training, MLflow registration, portability and rollback offline."""
    from mlops_sales_forecasting.aws.operations import rollback_model
    from mlops_sales_forecasting.aws.training_config import build_training_job_config
    from mlops_sales_forecasting.inference.bundle_loader import load_active_serving_bundle
    from mlops_sales_forecasting.inference.model_loader import load_xgboost_model
    from mlops_sales_forecasting.inference.releases.pointer import load_active_release_pointer
    from mlops_sales_forecasting.inference.releases.portable_publisher import (
        publish_portable_serving_release,
    )
    from mlops_sales_forecasting.orchestration import lifecycle_adapter
    from mlops_sales_forecasting.pipeline.project_factory import build_project_training_pipeline

    source, inputs = job_inputs
    config = build_training_job_config(
        source, input_directory=inputs, work_directory=tmp_path / "work"
    )
    config["tracking"] = {
        "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'shared.db'}",
        "experiment_name": "portable-lifecycle",
        "model_name": "portable-model",
        "promotion": {
            "metric_name": "rmse",
            "direction": "minimize",
            "minimum_improvement": 0,
            "allow_initial_champion": True,
        },
    }
    config["promotion"]["minimum_relative_rmse_improvement"] = 0
    client = MlflowClient(tracking_uri=config["tracking"]["mlflow_tracking_uri"])
    client.create_experiment("portable-lifecycle", artifact_location=(tmp_path / "mlflow").as_uri())
    old_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    # Keep Prefect orchestration outside this test, while exercising the real pipeline/lifecycle.
    monkeypatch.setattr(
        lifecycle_adapter,
        "run_prefect_training_pipeline",
        lambda *, pipeline, run_id: pipeline.run(run_id=run_id),
    )
    monkeypatch.setattr(
        lifecycle_adapter, "publish_serving_release", publish_portable_serving_release
    )

    def run():
        return lifecycle_adapter.run_prefect_model_lifecycle.fn(
            pipeline=build_project_training_pipeline(config),
        )

    try:
        first = run()
        assert first.candidate.registration.registered
        release1 = first.serving_release.manifest.release_id
        assert str(client.get_model_version_by_alias("portable-model", "champion").version) == "1"
        assert first.serving_release.manifest.model.uri.startswith(config["paths"]["models"])
        assert first.serving_release.manifest.model.version == "1"
        assert first.serving_release.manifest.metadata["registry_registered"] is True

        config["tracking"]["promotion"]["minimum_improvement"] = 1e9
        rejected = run()
        assert rejected.candidate.registration.model_version == "2"
        assert rejected.serving_release is None
        assert str(client.get_model_version_by_alias("portable-model", "challenger").version) == "2"
        assert (
            load_active_release_pointer(models_path=config["paths"]["models"]).release_id
            == release1
        )
        assert str(client.get_model_version_by_alias("portable-model", "champion").version) == "1"

        config["tracking"]["promotion"]["minimum_improvement"] = 0
        with monkeypatch.context() as failed:

            def fail_publication(**kwargs):
                raise OSError("simulated S3 publication failure")

            failed.setattr(lifecycle_adapter, "publish_serving_release", fail_publication)
            with pytest.raises(OSError, match="publication failure"):
                run()
        assert str(client.get_model_version_by_alias("portable-model", "champion").version) == "1"
        assert (
            load_active_release_pointer(models_path=config["paths"]["models"]).release_id
            == release1
        )

        next_result = run()
        release4 = next_result.serving_release.manifest.release_id
        assert str(client.get_model_version_by_alias("portable-model", "champion").version) == "4"
        # Remove the registry artifact store: serving and rollback use the portable release files.
        import shutil

        shutil.rmtree(tmp_path / "mlflow")
        bundle = load_active_serving_bundle(
            models_path=config["paths"]["models"],
            serving_alias="champion",
            model_loader=load_xgboost_model,
        )
        assert bundle.release_id == release4
        assert bundle.model.input_columns
        example = next_result.pipeline.pipeline.training.input_example
        np.testing.assert_allclose(
            bundle.model.predict(example),
            next_result.pipeline.pipeline.training.model.predict(example),
        )
        assert rollback_model(config=config, refresh=False) == release1
        assert str(client.get_model_version_by_alias("portable-model", "champion").version) == "1"
        assert (
            load_active_release_pointer(models_path=config["paths"]["models"]).release_id
            == release1
        )
        # Every portable model file is verified before deserialization.
        model_file = Path(first.serving_release.release_root) / "model/model.ubj"
        model_file.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="checksum"):
            load_active_serving_bundle(
                models_path=config["paths"]["models"],
                serving_alias="champion",
                model_loader=load_xgboost_model,
            )
    finally:
        mlflow.set_tracking_uri(old_uri)


def test_aws_simulation_reuses_clock_and_isolates_live_model(tmp_path, job_inputs, monkeypatch):
    """Exercise both arms, real training and logical-time retraining without AWS."""
    import shutil

    from mlops_sales_forecasting.aws import simulation
    from mlops_sales_forecasting.orchestration import lifecycle_adapter, retraining_service

    run_lifecycle = lifecycle_adapter.run_prefect_model_lifecycle.fn
    monkeypatch.setattr(simulation.cli, "run_prefect_model_lifecycle", run_lifecycle)
    monkeypatch.setattr(retraining_service, "run_prefect_model_lifecycle", run_lifecycle)
    monkeypatch.setattr(
        lifecycle_adapter,
        "run_prefect_training_pipeline",
        lambda *, pipeline, run_id: pipeline.run(run_id=run_id),
    )
    from mlops_sales_forecasting.data.raw.ingest import create_simulation_split

    config_path, inputs = job_inputs
    config = yaml.safe_load(config_path.read_text())
    config["environment"] = "dev"
    config["paths"]["raw_data"] = "s3://test-bucket/data/raw"
    config["tracking"]["mlflow_tracking_uri"] = f"sqlite:///{tmp_path / 'simulation.db'}"
    config["monitoring"]["retraining"].update(cooldown_hours=24, minimum_new_training_rows=1)
    config["monitoring"]["retraining"]["performance"]["minimum_samples"] = 1
    config["simulation"]["scheduled_interval_hours"] = 48
    config["simulation"]["retraining_triggers"] = {"scheduled_refresh": True}
    _, pool = create_simulation_split(pd.read_csv(inputs / "train.csv", parse_dates=["Date"]))
    pool.to_csv(inputs / "simulation_ground_truth.csv", index=False)
    destination = tmp_path / "uploaded"
    destination.mkdir()
    downloaded = []
    uploaded = []

    class S3:
        def download_file(self, bucket, key, filename):
            downloaded.append((bucket, key))
            shutil.copyfile(inputs / Path(key).name, filename)

        def upload_file(self, filename, bucket, key):
            uploaded.append((bucket, key))
            shutil.copyfile(filename, destination / Path(key).name)

    monkeypatch.setattr(simulation.boto3, "client", lambda service: S3())
    monkeypatch.setattr(simulation, "load_aws_lifecycle_config", lambda **kwargs: config)
    monkeypatch.chdir(tmp_path)
    previous_uri = mlflow.get_tracking_uri()
    try:
        report = simulation.run_aws_simulation(maximum_days=4)
        assert report["final_day"] == 4
        assert report["retraining_events"] >= 1
        assert report["model_name"] != config["tracking"]["model_name"]
        assert report["model_name"].endswith("-simulation")
        assert config["paths"]["raw_data"] == "s3://test-bucket/data/raw"
        assert len(downloaded) == 3
        assert all(key.startswith("artifacts/lifecycle_simulation/") for _, key in uploaded)
        assert json.loads((destination / "summary.json").read_text()) == report
        for name in ("with_retraining", "without_retraining"):
            assert len(pd.read_csv(destination / f"{name}.csv")) == 4
            assert (destination / f"{name}_evaluation.parquet").is_file()
        client = MlflowClient(tracking_uri=config["tracking"]["mlflow_tracking_uri"])
        assert not any(
            model.name == config["tracking"]["model_name"]
            for model in client.search_registered_models()
        )
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)
