import pandas as pd
import mlflow
import shutil
import logging
import warnings

from datetime import datetime
from google.cloud import storage

from src.configs.loader import get_path, load_config
from src.storage.filesystem import file_exists, ensure_dir

ENV_CFG = load_config()

# ruff: noqa: E402
from prefect import task, get_run_logger

from src.data.raw.ingest import ingest
from src.data.features.build_features import run_feature_pipeline
from src.data.features.create_state import create_feature_state
from src.data.features.calendar import (
    create_known_calendar_artifact,
)
from src.data.splits.split import split as split_logic
from src.data.versioning import make_dataset_version, snapshot_current_datasets, log_dataset_manifest_to_mlflow

from src.monitoring.drift import fetch_current_data, detect_ks_drift
from src.monitoring.feature_drift import run_feature_drift_check

from src.utils.logger import get_logger


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
    """
    Analyzes recent predictions against baseline training data.
    """
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


@task(name="Data Processing & Feature State Update")
def task_prepare_data(is_drift_run: bool):
    """
    Run ingestion, calendar creation, feature generation and dataset splitting.

    Args:
        is_drift_run: Whether data preparation is being performed in response to
            detected drift.

    Notes:
        Failure to update the inference feature state is logged but does not abort
        the remaining data preparation steps.
    """
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
    """
    Create an immutable snapshot of the currently prepared datasets.

    Returns:
        The dataset manifest containing the generated version identifier and
        snapshot locations.
    """
    p_logger = get_run_logger()
    version_id = make_dataset_version()
    manifest = snapshot_current_datasets(version_id)
    p_logger.info(f"Dataset snapshot created: {version_id}")
    return manifest

@task(name="Log Dataset Metadata")
def task_log_dataset_metadata(run_id: str, dataset_manifest: dict):
    """
    Attach a dataset manifest to an existing MLflow run.

    Args:
        run_id: MLflow run receiving the dataset metadata.
        dataset_manifest: Version and snapshot metadata to log.

    Notes:
        Logging failures are reported as warnings and do not fail the Prefect flow.
    """
    p_logger = get_run_logger()
    try:
        with mlflow.start_run(run_id=run_id):
            log_dataset_manifest_to_mlflow(dataset_manifest)
    except Exception as e:
        p_logger.warning(f"Could not log dataset metadata: {e}")

@task(name="Archive Logs")
def task_archive_logs():
    """
    Archives logs. Handles local files and now also GCS blobs.
    """

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

