# --- STANDARD LIBRARY IMPORTS ---
import sys
import os
import shutil
import logging
import warnings
import json

from datetime import datetime

# --- THIRD PARTY IMPORTS ---
import requests
import pandas as pd
import mlflow

from google.cloud import storage

# --- INTERNAL CONFIG BOOTSTRAP ---
from src.configs.loader import load_config, get_path, file_exists, ensure_dir, join_uri

# Load config early so environment variables (Prefect, MLflow) are set
ENV_CFG = load_config()

# --- PREFECT IMPORTS (after config bootstrap) ---
# ruff: noqa: E402
from prefect import flow, task, get_run_logger

# --- PROJECT IMPORTS ---
# ruff: noqa: E402
from src.data.raw.ingest import ingest
from src.data.features.build_features import run_feature_pipeline
from src.data.features.create_state import create_feature_state
from src.data.features.calendar import (
    create_known_calendar_artifact,
)
from src.data.splits.split import split as split_logic
from src.data.versioning import make_dataset_version, snapshot_current_datasets, log_dataset_manifest_to_mlflow

from src.deployment.verification import verify_serving_release

from src.training.train import train
from src.training.register import register_model
from src.training.evaluate import compare_models, evaluate_model, champion_exists
from src.training.policy import should_refresh_api, should_skip_training, get_run_strategy

from src.monitoring.drift import fetch_current_data, detect_ks_drift
from src.monitoring.feature_drift import run_feature_drift_check

from src.inference.serving_release import publish_serving_release

from src.utils.logger import get_logger

# --- INITIALIZE CONFIGURATION ---
GCP_CFG = load_config("gcp.yaml")
MODEL_NAME = ENV_CFG["model"]["registry_name"]
logger = get_logger(__name__)

# --- LOGGING SETUP ---
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

tracking_uri = ENV_CFG["tracking"]["mlflow_tracking_uri"]
mlflow.set_tracking_uri(tracking_uri)
logger.info(f"Using MLflow tracking URI: {tracking_uri}")

@task(name="Check Data Drift")
def task_check_drift():
    """Analyzes recent predictions against baseline training data."""
    p_logger = get_run_logger()
    curr_df = fetch_current_data() 
    if curr_df.empty:
        p_logger.info("No log data found for drift detection. Skipping check.")
        return False
    
    feature_drift_df = run_feature_drift_check()
    if not feature_drift_df.empty:
        drifted_features = feature_drift_df.loc[
            feature_drift_df["drift_detected"], "feature"
        ].tolist()

        p_logger.info(
            "Feature drift check completed | "
            f"drifted_features={drifted_features}"
        )
    else:
        p_logger.info("Feature drift check returned no results.")
        
    ref_file = f"{get_path('validated_data')}/train.parquet"
    if not file_exists(ref_file):
        p_logger.warning(f"Reference file {ref_file} missing. Cannot check drift.")
        return False
        
    ref_df = pd.read_parquet(ref_file)
    results = detect_ks_drift(ref_df["Sales"], curr_df["prediction"], column_name="Sales")
    
    p_logger.info(f"Drift Check Results: {results}")
    print(f"Drift status: {results['drift']}")
    return results["drift"]

@task(name="Evaluate Current Champion")
def task_evaluate_champion():
    p_logger = get_run_logger()
    p_logger.info("Evaluating current champion for dashboard continuity.")
    try:
        rmse = evaluate_model(model_alias="champion")
        print(f"Champion RMSE: {rmse}")
        return rmse
    except Exception as e:
        p_logger.warning(f"Could not evaluate champion: {e}")
        return None

@task(name="Data Processing & Feature State Update")
def task_prepare_data(is_drift_run: bool):
    p_logger = get_run_logger()

    p_logger.info(
        f"Starting data preparation "
        f"(Emergency Mode: {is_drift_run})"
    )

    ingest()

    p_logger.info(
        "Creating known calendar artifact."
    )
    create_known_calendar_artifact()

    run_feature_pipeline()

    p_logger.info(
        "Updating feature state snapshot for the API."
    )

    try:
        create_feature_state()
    except Exception as error:
        p_logger.error(
            f"Failed to update feature state: {error}"
        )

    split_logic(
        is_drift_run=is_drift_run
    )

@task(name="Snapshot Dataset Version")
def task_snapshot_dataset():
    p_logger = get_run_logger()
    version_id = make_dataset_version()
    manifest = snapshot_current_datasets(version_id)
    p_logger.info(f"Dataset snapshot created: {version_id}")
    return manifest

@task(name="Log Dataset Metadata")
def task_log_dataset_metadata(run_id: str, dataset_manifest: dict):
    p_logger = get_run_logger()
    try:
        with mlflow.start_run(run_id=run_id):
            log_dataset_manifest_to_mlflow(dataset_manifest)
    except Exception as e:
        p_logger.warning(f"Could not log dataset metadata: {e}")

@task(name="Model Training")
def task_train(is_drift_run: bool):
    p_logger = get_run_logger()

    p_logger.info(
        "Triggering model training task | drift_run=%s",
        is_drift_run,
    )

    model, run_id = train(
        is_drift_run=is_drift_run,
    )

    return run_id

@task(name="Candidate Evaluation")
def task_eval_and_reg(
    new_run_id: str,
) -> bool:
    """
    Evaluate the Candidate against the current Champion.

    The Candidate is accepted only when it passes every configured promotion
    gate. Rejected Candidates are retained in MLflow under the Challenger
    alias for audit, analysis and possible shadow evaluation.

    Comparison or policy errors propagate and block all registry changes.
    """
    p_logger = get_run_logger()

    candidate_accepted, metrics = (
        compare_models(
            new_run_id
        )
    )

    candidate_metrics = metrics.get(
        "candidate_metrics",
        {},
    )

    champion_metrics = metrics.get(
        "champion_metrics",
        {},
    )

    promotion_decision = metrics.get(
        "promotion_decision",
        {},
    )

    p_logger.info(
        "Promotion policy evaluated | "
        "candidate_run_id=%s | "
        "accepted=%s | "
        "candidate_rmse=%s | "
        "champion_rmse=%s | "
        "reasons=%s",
        new_run_id,
        promotion_decision.get(
            "accepted"
        ),
        candidate_metrics.get(
            "overall_rmse"
        ),
        champion_metrics.get(
            "overall_rmse"
        ),
        promotion_decision.get(
            "reasons",
            [],
        ),
    )

    # Compatibility output for the current lifecycle scripts.
    if "rmse_euro" in metrics:
        print(
            "Challenger RMSE: "
            f"{metrics['rmse_euro']}"
        )

    if candidate_accepted:
        p_logger.info(
            "Candidate passed all promotion gates | "
            f"candidate_run_id={new_run_id}"
        )
        return True

    p_logger.info(
        "Candidate rejected by promotion policy. "
        "Registering it as Challenger | "
        f"candidate_run_id={new_run_id} | "
        f"reasons={promotion_decision.get('reasons', [])}"
    )

    register_model(
        new_run_id,
        alias="challenger",
    )

    return False

@task(name="Bootstrap Initial Champion")
def task_bootstrap_champion(
    candidate_run_id: str,
    is_drift_run: bool,
) -> dict[str,str]:
    """
    Create the first Champion in an empty model registry.

    Bootstrap is rejected when a Champion already exists.
    """
    p_logger = get_run_logger()

    if champion_exists():
        raise RuntimeError(
            "Bootstrap rejected: a Champion already exists."
        )

    p_logger.info(
        "No Champion exists. Starting explicit initial bootstrap | "
        f"candidate_run_id={candidate_run_id}"
    )

    _, final_run_id = train(
        is_drift_run=is_drift_run,
        run_role="final_refit",
        candidate_run_id=candidate_run_id,
    )

    # Check again immediately before changing the alias.
    # This reduces the risk of two concurrent bootstrap runs.
    if champion_exists():
        raise RuntimeError(
            "Bootstrap aborted: a Champion was created concurrently."
        )

    model_version = register_model(
        final_run_id, 
        alias="champion",
    )

    p_logger.info(
        "Initial Champion created | "
        f"candidate_run_id={candidate_run_id} | "
        f"final_run_id={final_run_id}"
    )

    return {
        "run_id": final_run_id,
        "model_version": str(model_version.version),
    }
@task(name="Final Model Refit")
def task_final_refit(
    candidate_run_id: str,
    is_drift_run: bool,
) -> dict[str,str]:
    """
    Refit an accepted candidate on train and validation data.
    """
    p_logger = get_run_logger()

    p_logger.info(
        "Starting final refit | "
        f"candidate_run_id={candidate_run_id} | "
        f"drift_run={is_drift_run}"
    )

    _, final_run_id = train(
        is_drift_run=is_drift_run,
        run_role="final_refit",
        candidate_run_id=candidate_run_id,
    )

    model_version = register_model(
        final_run_id, 
        alias="champion",
    )

    p_logger.info(
        "Final refit registered as Champion | "
        f"final_run_id={final_run_id}"
    )

    return {
        "run_id": final_run_id,
        "model_version": str(model_version.version),
    }

@task(name="Publish Serving Release")
def task_publish_serving_release(
    *,
    final_run_id: str,
    model_version: str,
    dataset_manifest: dict,
) -> str:
    """
    Publish the promoted model and its inference artifacts as one immutable
    serving release.

    Works with local paths and gs:// paths.
    """
    p_logger = get_run_logger()

    client = mlflow.MlflowClient()
    run = client.get_run(
        final_run_id
    )

    model_type = (
        run.data.tags.get("model_type")
        or run.data.params.get("model_type")
        or "xgboost"
    )

    target_transformation = (
        run.data.tags.get(
            "target_transformation"
        )
        or run.data.params.get(
            "target_transformation"
        )
        or "none"
    )

    config_hash = run.data.params.get(
        "config_hash"
    )

    snapshots = dataset_manifest.get(
        "snapshots",
        {},
    )

    # Prefer the versioned dataset snapshot. This prevents the release from
    # reading store metadata that changed after this training run.
    store_metadata_source = snapshots.get(
        "validated_store"
    )

    if not store_metadata_source:
        store_metadata_source = join_uri(
            get_path("validated_data"),
            "store.parquet",
        )

    store_state_source = join_uri(
        get_path("models"),
        "latest_state.json",
    )

    known_calendar_source = join_uri(
        get_path("features"),
        "known_calendar.parquet",
    )

    manifest = publish_serving_release(
        models_path=get_path("models"),
        model_name=MODEL_NAME,
        model_version=model_version,
        model_run_id=final_run_id,
        model_type=model_type,
        target_transformation=(
            target_transformation
        ),
        dataset_version=(
            dataset_manifest.get(
                "dataset_version"
            )
        ),
        config_hash=config_hash,
        git_commit=(
            dataset_manifest.get(
                "git_commit"
            )
            or os.getenv(
                "GIT_COMMIT_SHA"
            )
        ),
        store_metadata_source=(
            store_metadata_source
        ),
        store_state_source=(
            store_state_source
        ),
        known_calendar_source=(
            known_calendar_source
        ),
    )

    p_logger.info(
        "Serving release published | "
        f"release_id={manifest.release_id} | "
        f"model_version={model_version} | "
        f"dataset_version={manifest.dataset_version}"
    )

    return manifest.release_id


@task(name="Archive Logs")
def task_archive_logs():
    """Archives logs. Handles local files and now also GCS blobs."""

    archived_count = 0
    try:
        p_logger = get_run_logger()
    except Exception:
        p_logger = logger

    PREDICTIONS_PATH = get_path("predictions")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- GCS ARCHIVING LOGIC ---
    if PREDICTIONS_PATH.startswith("gs://"):
        try:
            # Parse bucket and folder
            path_no_gs = PREDICTIONS_PATH.replace("gs://", "")
            bucket_name = path_no_gs.split("/")[0]
            source_folder = "/".join(path_no_gs.split("/")[1:])
            if source_folder and not source_folder.endswith("/"):
                source_folder += "/"
            
            archive_folder = f"{source_folder}archive/"
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=source_folder)
            
            archived_count = 0
            for blob in blobs:
                # Skip the directory placeholders and anything already in archive
                if blob.name == source_folder or "archive/" in blob.name:
                    continue
                
                filename = blob.name.split("/")[-1]
                new_blob_name = f"{archive_folder}{timestamp}_{filename}"
                
                # Move = Copy + Delete
                bucket.copy_blob(blob, bucket, new_blob_name)
                blob.delete()
                archived_count += 1
            
            p_logger.info(f"GCS: Successfully archived {archived_count} files to {archive_folder}")
        except Exception as e:
            p_logger.error(f"Failed to archive GCS logs: {e}")

    # --- LOCAL ARCHIVING LOGIC ---
    else:
        log_file = f"{PREDICTIONS_PATH}/inference_log.parquet"
        if file_exists(log_file):
            archive_dir = f"{PREDICTIONS_PATH}/archive"
            ensure_dir(archive_dir)
            target_path = f"{archive_dir}/inference_log_{timestamp}.parquet"
            shutil.move(log_file, target_path)
            p_logger.info(f"Local: Logs archived to: {target_path}")
        else:
            p_logger.info("Local: No log file found to archive.")
    
    return archived_count

@task(name="Refresh API")
def task_refresh_api() -> None:
    """
    Refresh the forecasting API serving state after training.

    Reloads:
    - current champion model
    - store metadata
    - forecasting state snapshot
    """
    p_logger = get_run_logger()
    cfg = load_config()

    api_url = cfg.get("api", {}).get("url", "http://api:8080/predict")

    if api_url.endswith("/predict"):
        base_url = api_url.removesuffix("/predict")
    else:
        base_url = api_url.rstrip("/")

    reload_url = f"{base_url}/admin/reload-serving-state"

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is not set.")

    p_logger.info(f"Refreshing API serving state via: {reload_url}")

    response = requests.post(
        reload_url,
        headers={"X-API-KEY": api_key},
        timeout=30,
    )

    response.raise_for_status()
    p_logger.info(f"API serving state reload successful: {response.json()}")

    
@task(name="Verify Serving Release")
def task_verify_serving_release(
    expected_release_id: str,
) -> dict:
    """
    Verify that the API activated the expected complete serving release.
    """

    p_logger = get_run_logger()

    api_url = ENV_CFG.get(
        "api",
        {},
    ).get(
        "url",
        "http://api:8080/predict",
    )

    if api_url.endswith("/predict"):
        api_base_url = (
            api_url.removesuffix(
                "/predict"
            )
        )
    else:
        api_base_url = api_url.rstrip("/")

    p_logger.info(
        "Verifying serving release | "
        f"expected_release_id="
        f"{expected_release_id} | "
        f"api_base_url={api_base_url}"
    )

    result = verify_serving_release(
        api_base_url=api_base_url,
        expected_release_id=(
            expected_release_id
        ),
    )

    p_logger.info(
        "Serving release verified | "
        f"release_id="
        f"{result.release_id} | "
        f"model_version="
        f"{result.model_version} | "
        f"model_run_id="
        f"{result.model_run_id} | "
        f"attempts={result.attempts}"
    )

    return {
        "release_id": result.release_id,
        "model_version": (
            result.model_version
        ),
        "model_run_id": (
            result.model_run_id
        ),
        "attempts": result.attempts,
    }


@flow(name="End-to-End Demand Forecasting Pipeline")
def training_pipeline(
    force_run: bool = False,
    bootstrap: bool = False,
):
    if bootstrap and champion_exists():
        raise RuntimeError(
            "Bootstrap rejected: a Champion already exists. "
            "Use the regular forced training flow instead."
        )
    
    p_logger = get_run_logger()
    p_logger.info(f"Starting Pipeline (Env: {ENV_CFG['environment']})")
    
    drift_detected = task_check_drift()

    if should_skip_training(drift_detected, force_run):
        p_logger.info("System stable. Only evaluating current champion.")
        task_evaluate_champion()
        return

    strategy = get_run_strategy(drift_detected, force_run)
    print(f"[{strategy}] mode activated.")
    
    task_prepare_data(is_drift_run=drift_detected)
    dataset_manifest = task_snapshot_dataset()
    run_id = task_train(is_drift_run=drift_detected)
    task_log_dataset_metadata(run_id, dataset_manifest)

    #task_archive_logs() 

    serving_run_id = run_id
    serving_model_version = None
    release_id = None
    candidate_accepted = False
    new_champion_crowned = False

    if bootstrap:
        promotion_result = (
            task_bootstrap_champion(
                candidate_run_id=run_id,
                is_drift_run=drift_detected,
            )
        )

        serving_run_id = (
            promotion_result["run_id"]
        )
        serving_model_version = (
            promotion_result["model_version"]
        )

        task_log_dataset_metadata(
            serving_run_id,
            dataset_manifest,
        )

        release_id = (
            task_publish_serving_release(
                final_run_id=serving_run_id,
                model_version=(
                    serving_model_version
                ),
                dataset_manifest=(
                    dataset_manifest
                ),
            )
        )

        candidate_accepted = True
        new_champion_crowned = True

    else:
        candidate_accepted = task_eval_and_reg(
            run_id
        )

        if candidate_accepted:
            promotion_result = (
                task_final_refit(
                    candidate_run_id=run_id,
                    is_drift_run=drift_detected,
                )
            )

            serving_run_id = (
                promotion_result["run_id"]
            )
            serving_model_version = (
                promotion_result["model_version"]
            )

            task_log_dataset_metadata(
                serving_run_id,
                dataset_manifest,
            )

            release_id = (
                task_publish_serving_release(
                    final_run_id=serving_run_id,
                    model_version=(
                        serving_model_version
                    ),
                    dataset_manifest=(
                        dataset_manifest
                    ),
                )
            )

            new_champion_crowned = True

    if should_refresh_api(
        new_champion_crowned
    ):
        if not release_id:
            raise RuntimeError(
                "Champion was promoted but no serving "
                "release was published."
            )

        p_logger.info(
            "New serving release published. "
            "Refreshing API | "
            f"release_id={release_id}"
        )

        task_refresh_api()
        task_verify_serving_release(
            expected_release_id=release_id,
        )

    else:
        p_logger.info(
            "No API refresh needed. "
            "Current serving release remains active."
        )
        
    p_logger.info("Pipeline execution finished successfully.")

    return {
        "run_id": serving_run_id,
        "candidate_run_id": run_id,
        "final_refit_run_id": (
            serving_run_id
            if candidate_accepted
            else None
        ),
        "model_version": (
            serving_model_version
        ),
        "release_id": release_id,
        "champion_promoted": bool(
            new_champion_crowned
        ),
    }

if __name__ == "__main__":
    force = "--force" in sys.argv
    bootstrap = "--bootstrap" in sys.argv

    result = training_pipeline(
        force_run=force,
        bootstrap=bootstrap,
    )
    print("TRAINING_RESULT_JSON=" + json.dumps(result))
    