import pandas as pd
import os
import glob
import gcsfs
from datetime import datetime
from src.configs.loader import load_config, get_path
from src.data.validation.validate import validate_train
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load configuration
CFG = load_config()
GCP_CFG = load_config("gcp.yaml")

def simulate_ground_truth_injection(drift_factor=1):
    """
    Simulates daily data injection by moving one day from the simulation pool 
    to the active 'new_batches' directory. Applies drift after Day 5.
    """
    source_path = os.path.join(get_path("raw_data"), "simulation_ground_truth.csv")
    target_dir = os.path.join(get_path("raw_data"), "new_batches")
    
    # Check if we are operating on Google Cloud Storage or local filesystem
    is_gcs = source_path.startswith("gs://")
    fs = gcsfs.GCSFileSystem() if is_gcs else None

    # 1. Existence Check: Standard os.path doesn't support gs://
    exists = fs.exists(source_path) if is_gcs else os.path.exists(source_path)
    
    if not exists:
        logger.error(f"Simulation source not found at {source_path}. Run ingest.py first.")
        return

    # 2. Load the pool: Pandas uses gcsfs internally for gs:// paths
    try:
        df = pd.read_csv(source_path, parse_dates=["Date"], dtype={"StateHoliday": str})
    except Exception as e:
        logger.error(f"Failed to read simulation pool: {e}")
        return

    if df.empty:
        logger.info("Simulation pool is empty.")
        print("Remaining days in pool: 0")
        return

    # 3. Determine current simulation day to decide on drift application
    # We count existing files in the target directory to track progress
    if is_gcs:
        # Extract bucket name and path for GCS globbing
        bucket_name = GCP_CFG["gcp"]["gcs"]["bucket_name"]
        search_pattern = f"gs://{bucket_name}/data/raw/new_batches/ground_truth_*.csv"
        existing_batches = fs.glob(search_pattern)
    else:
        os.makedirs(target_dir, exist_ok=True)
        existing_batches = glob.glob(os.path.join(target_dir, "ground_truth_*.csv"))
    
    current_day_index = len(existing_batches) + 1
    
    # Logic: Day 1-5 stable (factor 1.0), Day 6+ applies drift_factor
    active_drift = 1.0 if current_day_index <= 5 else drift_factor
    
    if active_drift == 1.0:
        logger.info(f"Day {current_day_index}: Simulating STABLE data (No drift).")
    else:
        logger.warning(f"Day {current_day_index}: Simulating DRIFT (Factor: {active_drift}).")

    # 4. Extract the next chronological date for simulation
    unique_dates = sorted(df["Date"].unique())
    next_date = unique_dates[0]
    batch_data = df[df["Date"] == next_date].copy()
    
    logger.info(f"Processing Day {current_day_index} for date: {next_date.date()}")

    # 5. Apply drift to Sales column
    batch_data["Sales"] = (batch_data["Sales"] * active_drift).astype(int)

    # 6. Validate the generated batch
    try:
        validate_train(batch_data)
    except Exception as e:
        logger.error(f"Validation failed for batch: {e}")
        return

    # 7. Save and Upload the new batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_filename = f"ground_truth_{timestamp}.csv"

    if is_gcs:
        # Write directly to Cloud Storage
        dest_bucket = GCP_CFG["gcp"]["gcs"]["bucket_name"]
        cloud_target = f"gs://{dest_bucket}/data/raw/new_batches/{batch_filename}"
        batch_data.to_csv(cloud_target, index=False)
        logger.info(f"Cloud: Uploaded batch to GCS: {batch_filename}")
    else:
        # Save to local filesystem
        batch_path = os.path.join(target_dir, batch_filename)
        batch_data.to_csv(batch_path, index=False)
        logger.info(f"Local: Saved batch: {batch_filename}")

    # 8. Update simulation pool by removing the processed day
    remaining_pool = df[df["Date"] > next_date]
    remaining_pool.to_csv(source_path, index=False)
    
    num_remaining = remaining_pool['Date'].nunique()
    logger.info(f"Remaining days in pool: {num_remaining}")
    
    # Print statement for the run_drift_demo.py to capture progress
    print(f"Remaining days in pool: {num_remaining}")

if __name__ == "__main__":
    simulate_ground_truth_injection()