import mlflow
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from src.configs.loader import get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_baseline_comparison():
    results_file = "evolution_results_90days_baseline.csv"
    if not os.path.exists(results_file):
        logger.error("No evolution_results_90days_baseline.csv found!")
        return

    df_results = pd.read_csv(results_file)
    
    # 1. Load the INITIAL Champion (Version 1)
    # This is the model before any automation kicked in
    model_name = "demand-forecasting-model-dev" # Check if this matches your CFG
    try:
        # We load version 1 specifically to represent the "static" approach
        static_model = mlflow.xgboost.load_model(f"models:/{model_name}/1")
        logger.info("Loaded Static Model (Version 1) for comparison.")
    except Exception as e:
        logger.error(f"Could not load Version 1: {e}")
        return

    static_rmses = []

    # 2. Iterate through the days recorded in your results
    for index, row in df_results.iterrows():
        day = int(row['day'])
        
        # We need the validation data that the API saw on that specific day
        # If you don't save val_data per day, we use the current one as a proxy
        # for the 'drifted' state.
        val_path = f"{get_path('splits')}/val.parquet" 
        
        try:
            val_df = pd.read_parquet(val_path)
            X_val = val_df.drop(columns=["Sales", "Date"], errors='ignore')
            y_val = val_df["Sales"]

            # --- THE CORE PART: PREDICTION ---
            # Most likely your V1 also used log1p, so we transform back
            preds_log = static_model.predict(X_val)
            preds_euro = np.expm1(preds_log)
            
            # Calculate what the error WOULD have been
            rmse = np.sqrt(mean_squared_error(y_val, preds_euro))
            static_rmses.append(rmse)
            
        except Exception as e:
            logger.warning(f"Could not calculate baseline for day {day}: {e}")
            static_rmses.append(None)

    # 3. Save back to the CSV
    df_results['static_rmse_euro'] = static_rmses
    df_results.to_csv(results_file, index=False)
    logger.info("Successfully added static_rmse_euro to evolution_results.csv")

if __name__ == "__main__":
    run_baseline_comparison()