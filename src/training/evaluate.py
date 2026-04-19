import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from mlflow.tracking import MlflowClient
from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger
from src.training.utils import build_drop_columns
from src.training.target_transform import inverse_transform_target

# Initialize project-specific logger
# English comments for consistency
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
        
        preds = model.predict(X_val)
        
        # Check for log scale
        if run.data.tags.get("target_transformation") == "log1p" or \
           run.data.params.get("target_transformation") == "log1p":
            preds = np.expm1(preds)
            
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return float(rmse)
    except Exception as e:
        logger.warning(f"Could not evaluate {model_alias}: {e}")
        return None

def compare_models(new_run_id: str, val_path: str | None = None):
    """
    Compares the new model (Challenger) with the current best model (Champion).
    Returns (is_better: bool, metrics: dict).
    """
    client = MlflowClient()
    
    # 1. Load the current validation data (Real scale)
    if val_path is None: 
        val_path = f"{get_path('splits')}/val.parquet"
        
    logger.info(f"Loading validation data for model comparison: {val_path}")
    drop_columns = build_drop_columns(TRAIN_CFG)
    try:
        val_df = pd.read_parquet(val_path)
        X_val = val_df.drop(columns=drop_columns, errors="ignore")
        y_val = val_df[TRAIN_CFG["data"]["target_column"]]
    except Exception as e:
        logger.error(f"Failed to load validation data: {e}")
        raise e
    
    # 2. Evaluate the Challenger
    logger.info(f"Evaluating Challenger (Run ID: {new_run_id})...")
    challenger_uri = f"runs:/{new_run_id}/model"
    challenger = mlflow.xgboost.load_model(challenger_uri)
    
    challenger_run = client.get_run(new_run_id)
    challenger_transform = (
        challenger_run.data.tags.get("target_transformation")
        or challenger_run.data.params.get("target_transformation")
        or "none"
    )

    chall_preds = challenger.predict(X_val)
    chall_preds = inverse_transform_target(chall_preds, challenger_transform)

    chall_rmse = np.sqrt(mean_squared_error(y_val, chall_preds))
    metrics = {"challenger_rmse": chall_rmse, "rmse_euro": chall_rmse}

    # 3. Evaluate the current Champion
    try:
        champion_uri = f"models:/{MODEL_NAME}@champion"
        logger.info(f"Evaluating current Champion from Registry: {champion_uri}")
        
        latest_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champ_run_id = latest_version.run_id
        champion = mlflow.xgboost.load_model(champion_uri)
        
        run_info = client.get_run(champ_run_id)
        if run_info.data.tags.get("target_transformation") == "log1p" or \
            run_info.data.params.get("target_transformation") == "log1p":
             champ_preds = np.expm1(champion.predict(X_val))
        else:
             champ_preds = champion.predict(X_val)

        champ_rmse = np.sqrt(mean_squared_error(y_val, champ_preds))
        metrics["champion_rmse"] = champ_rmse
        
        logger.info("--- Fair 'Real-Scale' Comparison ---")
        logger.info(f" -> Challenger RMSE: {chall_rmse:.4f}")
        logger.info(f" -> Champion RMSE:   {champ_rmse:.4f}")

        is_better = chall_rmse < champ_rmse
        return is_better, metrics

    except Exception as e:
        logger.warning(f"Comparison skipped (Reason: {e}). Challenger wins by default.")
        return True, metrics

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default_run_id"
    compare_models(run_id)