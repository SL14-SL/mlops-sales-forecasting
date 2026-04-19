import os
import pandas as pd
from scipy.stats import ks_2samp
from google.cloud import bigquery
from src.configs.loader import load_config, get_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CFG = load_config()
GCP_CFG = load_config("gcp.yaml")
PREDICTIONS = get_path("predictions")

def fetch_current_data():
    """
    Fetches the latest prediction logs from BigQuery (prod) or local parquet (dev).
    """
    if os.getenv("APP_ENV") == "prod":
        client = bigquery.Client()
        project_id = GCP_CFG["gcp"]["project_id"]
        
        query = f"""
            SELECT 
                jsonPayload.input.store,
                jsonPayload.input.date,
                jsonPayload.input.customers,
                jsonPayload.input.promo,
                jsonPayload.prediction
            FROM `{project_id}.logs.run_googleapis_com_stdout_*`
            WHERE jsonPayload.prediction IS NOT NULL
        """
        try:
            df = client.query(query).to_dataframe()
            df.columns = ["Store", "Date", "Customers", "Promo", "prediction"]
            return df
        except Exception as e:
            print(f"❌ BigQuery fetch failed: {e}")
            return pd.DataFrame()
    else:
        log_file = os.path.join(PREDICTIONS, "inference_log.parquet")
        if os.path.exists(log_file):
            return pd.read_parquet(log_file)
        return pd.DataFrame()

def detect_ks_drift(reference_data, current_data, column_name, threshold=0.01):
    """
    Performs KS-test with robustness checks.
    1. Increased strictness (threshold 0.01) to avoid flickering alarms.
    2. Added a minimum 'effect size' check (statistic > 0.1).
    3. Filters out zeros (closed days/empty logs).
    """
    # Filter out zeros to avoid bias from closed days
    ref = reference_data[reference_data > 0].dropna()
    cur = current_data[current_data > 0].dropna()

    if len(cur) < 50: # Need a minimum sample size for statistical power
        return {
            "feature": column_name, 
            "drift": False, 
            "p_value": 1.0, 
            "statistic": 0.0, 
            "samples": len(cur)
        }

    stat, p_value = ks_2samp(ref, cur)
    
    # Drift is only True if:
    # - p_value is very low (statistical significance)
    # - AND the distance between distributions is meaningful (> 0.1)
    # This prevents tiny, irrelevant distribution shifts from triggering a full retrain.
    is_drift = (p_value < threshold) and (stat > 0.1)
    
    return {
        "feature": column_name,
        "drift": bool(is_drift),
        "p_value": float(p_value),
        "statistic": float(stat),
        "samples": len(cur)
    }

