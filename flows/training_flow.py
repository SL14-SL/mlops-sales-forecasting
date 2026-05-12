# --- STANDARD LIBRARY IMPORTS ---
import sys
import os
import time
import shutil
import logging
import warnings

from datetime import datetime

# --- THIRD PARTY IMPORTS ---
import requests
import pandas as pd
import mlflow

from google.cloud import storage

# --- INTERNAL CONFIG BOOTSTRAP ---
from src.configs.loader import load_config, get_path, file_exists, ensure_dir

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
from src.data.splits.split import split as split_logic
from src.data.versioning import make_dataset_version, snapshot_current_datasets, log_dataset_manifest_to_mlflow

from src.training.train import train
from src.training.register import register_model
from src.training.evaluate import compare_models, evaluate_model
from src.training.policy import should_refresh_api, should_skip_training, get_run_strategy

from src.monitoring.drift import fetch_current_data, detect_ks_drift
from src.monitoring.feature_drift import run_feature_drift_check

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
    p_logger.info(f"Starting data preparation (Emergency Mode: {is_drift_run})")
    ingest()
    run_feature_pipeline()
    p_logger.info("Updating feature state snapshot for the API.")
    try:
        create_feature_state()
    except Exception as e:
        p_logger.error(f"Failed to update feature state: {e}")
    split_logic(is_drift_run=is_drift_run)

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
def task_train():
    p_logger = get_run_logger()
    p_logger.info("Triggering model training task.")
    model, run_id = train()
    return run_id

@task(name="Evaluation & Registration")
def task_eval_and_reg(new_run_id: str):
    p_logger = get_run_logger()
    is_better, metrics = compare_models(new_run_id)

    if metrics and "rmse_euro" in metrics:
        print(f"Challenger RMSE: {metrics['rmse_euro']}")

    if is_better:
        p_logger.info(f"Challenger (Run: {new_run_id}) outperforms Champion. Promoting...")
        register_model(new_run_id, alias="champion")
        return True
    else:
        p_logger.info("Champion remains superior. Registering new model as Challenger.")
        register_model(new_run_id, alias="challenger")
        return False


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
    Refresh the API model after a new champion has been promoted.

    Calls the API reload endpoint instead of restarting the container.
    """
    p_logger = get_run_logger()
    cfg = load_config()

    api_url = cfg.get("api", {}).get("url", "http://api:8080/predict")
    base_url = api_url.replace("/predict", "")
    reload_url = f"{base_url}/admin/reload-model"

    api_key = os.getenv("API_KEY")

    response = requests.post(
        reload_url,
        headers={"X-API-KEY": api_key},
        timeout=30,
    )

    response.raise_for_status()
    p_logger.info(f"API model reload successful: {response.json()}")

@task(name="Verify API Health")
def task_verify_health():
    p_logger = get_run_logger()

    api_url = ENV_CFG["api"]["url"].rsplit("/", 1)[0] + "/health"
    p_logger.info(f"Checking API health at: {api_url}")

    for i in range(20):  # 20 Versuche
        try:
            r = requests.get(api_url, timeout=10)

            if r.status_code == 200:
                p_logger.info("API is healthy.")
                return True

            p_logger.warning(f"Attempt {i+1}: status {r.status_code}")

        except requests.exceptions.RequestException as e:
            p_logger.warning(f"Attempt {i+1}: not reachable ({e})")

        time.sleep(10)

    raise Exception("API did not recover after refresh!")


@flow(name="End-to-End Demand Forecasting Pipeline")
def training_pipeline(force_run: bool = False):
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
    run_id = task_train()
    task_log_dataset_metadata(run_id, dataset_manifest)

    #task_archive_logs() 

    new_champion_crowned = task_eval_and_reg(run_id)
    if should_refresh_api(new_champion_crowned):
        p_logger.info("🚀 New Champion detected. Refreshing API...")
        task_refresh_api()
        task_verify_health()
    else:
        p_logger.info("✅ No API refresh needed. Current Champion is still the best.")

    p_logger.info("Pipeline execution finished successfully.")

    return {
        "run_id": run_id,
        "champion_promoted": bool(new_champion_crowned),
    }

if __name__ == "__main__":
    force = "--force" in sys.argv
    import json

    result = training_pipeline(force_run=force)
    print("TRAINING_RESULT_JSON=" + json.dumps(result))
    