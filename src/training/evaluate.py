import mlflow
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

from mlflow.tracking import MlflowClient

from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger
from src.training.utils import build_drop_columns

from src.training.evaluate_metrics import align_features_for_evaluation
from src.training.model_comparison import compare_models

# Initialize project-specific logger
logger = get_logger(__name__)

# Load central config
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]



def evaluate_model(model_alias: str = "champion") -> float:
    """
    Evaluates a specific model from the registry (e.g., 'champion') 
    on the current validation set and returns the RMSE in Euro scale.
    """
    client = MlflowClient()
    
    # 1. Load validation data
    val_path = f"{get_path('splits')}/val.parquet"
    drop_columns = build_drop_columns(TRAIN_CFG)
    try:
        val_df = pd.read_parquet(val_path)
        X_val = val_df.drop(columns=drop_columns, errors="ignore")
        y_val = val_df[TRAIN_CFG["data"]["target_column"]]
    except Exception as e:
        logger.error(f"Failed to load validation data: {e}")
        return None

    # 2. Load model from registry
    try:
        model_uri = f"models:/{MODEL_NAME}@{model_alias}"
        model = mlflow.xgboost.load_model(model_uri)
        
        # Get run info to check for log transformation
        version = client.get_model_version_by_alias(MODEL_NAME, model_alias)
        run = client.get_run(version.run_id)
        
        aligned_X_val = align_features_for_evaluation(
            model,
            X_val,
        )

        preds = model.predict(aligned_X_val)    

        # Check for log scale
        if run.data.tags.get("target_transformation") == "log1p" or \
           run.data.params.get("target_transformation") == "log1p":
            preds = np.expm1(preds)
            
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return float(rmse)
    except Exception as e:
        logger.warning(f"Could not evaluate {model_alias}: {e}")
        return None



if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default_run_id"
    compare_models(run_id)