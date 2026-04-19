import pandas as pd
import os
import shutil
import gcsfs
from src.configs.loader import load_config, get_path
from src.data.validation.validate import validate_store, validate_train
from src.utils.logger import get_logger

# Initialize project-specific logger
logger = get_logger(__name__)

def ingest(): 
    """
    Main ingestion task: 
    - Loads raw data and validates it.
    - Performs a chronological 90/10 split for simulation.
    - Merges new batches with individual validation and quarantine logic.
    """
    # Resolve paths inside the function for better testability (Late Binding)
    #cfg = load_config()
    gcp_cfg = load_config("gcp.yaml")
    raw_path = get_path("raw_data")
    validated_path = get_path("validated_data")
    
    env = os.getenv("APP_ENV", "dev")
    logger.info(f"Starting ingestion process. Source: {raw_path} | Env: {env}")
    
    # 1. Load original Kaggle source files
    try:
        train_full = pd.read_csv(f"{raw_path}/train.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
        store = pd.read_csv(f"{raw_path}/store.csv")
        logger.info("Base source files loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load base source files: {e}")
        return

    # 2. Initial Validation
    validate_store(store)
    validate_train(train_full)
    logger.info("Initial validation of base data passed.")


    # 3. Chronological 90/10 Split on unique dates
    train_full = train_full.sort_values("Date", ascending=True).copy()
    train_full["Date"] = pd.to_datetime(train_full["Date"], errors="coerce")

    if train_full["Date"].isna().any():
        raise ValueError("Found invalid dates in training data during chronological split.")

    unique_dates = sorted(train_full["Date"].dropna().unique())
    date_split_idx = int(len(unique_dates) * 0.9)
    date_split_idx = min(max(date_split_idx, 0), len(unique_dates) - 1)

    split_date = pd.Timestamp(unique_dates[date_split_idx])

    train_base = train_full[train_full["Date"] < split_date].copy()
    sim_truth = train_full[train_full["Date"] >= split_date].copy()

    logger.info(
        "Chronological date-based split completed | "
        f"split_date={split_date.date()} | "
        f"train_rows={len(train_base)} | sim_rows={len(sim_truth)} | "
        f"train_max_date={train_base['Date'].max().date() if not train_base.empty else None} | "
        f"sim_min_date={sim_truth['Date'].min().date() if not sim_truth.empty else None}"
    )
    
    # Save simulation source if missing
    sim_source_path = f"{raw_path}/simulation_ground_truth.csv"
    
    # Check if file exists (works for local and GCS)
    file_exists = False
    if raw_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        file_exists = fs.exists(sim_source_path)
    else:
        file_exists = os.path.exists(sim_source_path)

    if not file_exists:
        # Pandas can write directly to gs:// if gcsfs is installed
        sim_truth.to_csv(sim_source_path, index=False)
        logger.info(f"Created simulation source: {sim_source_path}")
    # 4. Collect Incremental Batches with Quarantine Logic
    final_train = train_base
    new_batches_found = []
    
    batch_dir = f"{raw_path}/new_batches"
    quarantine_dir = f"{raw_path}/quarantine"

    # Cloud Ingestion (GCS)
    if env == "prod" or raw_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        bucket_name = gcp_cfg["gcp"]["gcs"]["bucket_name"]
        batch_pattern = f"gs://{bucket_name}/data/raw/new_batches/*.csv"
        try:
            batch_files = fs.glob(batch_pattern)
            for f in batch_files:
                try:
                    # Using gsfs to read cloud batches
                    batch_df = pd.read_csv(f"gs://{f}", parse_dates=["Date"], dtype={"StateHoliday": str})
                    validate_train(batch_df)
                    new_batches_found.append(batch_df)
                except Exception as e:
                    logger.warning(f"Cloud Batch '{f}' failed validation: {e}")
        except Exception as e:
            logger.error(f"Error accessing GCS bucket: {e}")
            pass 
    
    # Local Ingestion
    else:
        if os.path.exists(batch_dir):
            if not os.path.exists(quarantine_dir):
                os.makedirs(quarantine_dir)

            for file in os.listdir(batch_dir):
                if file.endswith(".csv"):
                    batch_path = os.path.join(batch_dir, file)
                    try:
                        batch_df = pd.read_csv(batch_path, parse_dates=["Date"], dtype={"StateHoliday": str})
                        validate_train(batch_df)
                        new_batches_found.append(batch_df)
                        logger.info(f"Batch '{file}' validated successfully.")

                    except Exception as e:
                        logger.warning(f"Batch '{file}' rejected: {e}")
                        dest_path = os.path.join(quarantine_dir, file)
                        shutil.move(batch_path, dest_path)
                        logger.info(f"Moved corrupted file '{file}' to quarantine.")

    # 5. Final Merge & Re-Validation
    if new_batches_found:
        final_train = pd.concat([train_base] + new_batches_found, ignore_index=True)
        final_train = final_train.sort_values("Date", ascending=True)
        validate_train(final_train)
        logger.info(f"Integrated {len(new_batches_found)} new batches into training set.")

    # Ensure output directory exists
    if not validated_path.startswith("gs://"):
        os.makedirs(validated_path, exist_ok=True)

    # 6. Export to Parquet
    final_train.to_parquet(f"{validated_path}/train.parquet", index=False)
    store.to_parquet(f"{validated_path}/store.parquet", index=False)
    
    logger.info(f"Ingestion complete. Total rows: {len(final_train)} | Output: {validated_path}")

if __name__ == "__main__":
    ingest()