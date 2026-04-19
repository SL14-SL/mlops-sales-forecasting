import pandas as pd
import os
from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load environment config
CFG = load_config()

FEATURES = get_path("features")
SPLITS = get_path("splits")

def split(is_drift_run: bool = False):
    """
    Implements a sliding window approach with dynamic validation sizing.
    If a drift is detected, we use an 'Emergency Window' (2 days) 
    to force new patterns into the training set faster.
    """
    input_file = f"{FEATURES}/features.parquet"
    
    logger.info(f"Loading features for splitting from: {input_file}")
    df = pd.read_parquet(input_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Define the overall window (last 182 days / 6 months)
    max_date = df["Date"].max()
    min_date = df["Date"].min()
    start_date = min_date
    # start_date = max_date - pd.Timedelta(days=182)
    
    # Filter for recent data only to avoid obsolete patterns
    df = df[df["Date"] >= start_date].copy()
    logger.info(f"Sliding window range: {start_date.date()} to {max_date.date()}")

    # 2. Dynamic Validation Window
    # Shrink window during drift to accelerate learning from new patterns
    if is_drift_run:
        logger.warning("Drift detected! Applying Emergency Window logic.")
        # Emergency logic: Ensure we have at least 2 open days for validation
        potential_val_data = df[df["Date"] > (max_date - pd.Timedelta(days=7))]
        open_days = potential_val_data[potential_val_data["Open"] == 1]["Date"].unique()
        open_days = sorted(open_days, reverse=True)
        
        if len(open_days) >= 2:
            cutoff_date = open_days[1] 
        else:
            cutoff_date = max_date - pd.Timedelta(days=2)
    else:
        logger.info("Applying normal validation window (14 days).")
        cutoff_date = max_date - pd.Timedelta(days=14)

    train = df[df["Date"] < cutoff_date]
    val = df[df["Date"] >= cutoff_date]

    # Save the splits
    os.makedirs(SPLITS, exist_ok=True)
    train.to_parquet(f"{SPLITS}/train.parquet", index=False)
    val.to_parquet(f"{SPLITS}/val.parquet", index=False)
    
    status = "EMERGENCY" if is_drift_run else "NORMAL"
    train_start = train["Date"].min().date()
    train_end = train["Date"].max().date()
    val_days = val["Date"].nunique()
    
    # Final summary of the split operation
    logger.info(f"[{status}] Data split complete.")
    logger.info(f"[{status}] Train set: {train_start} to {train_end} ({len(train)} rows)")
    logger.info(f"[{status}] Val set: {val_days} days starting {val['Date'].min().date()} ({len(val)} rows)")

if __name__ == "__main__":
    # Default to normal split if called directly
    split(is_drift_run=False)