import mlflow
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

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

class ModelComparisonError(RuntimeError):
    """Raised when a safe Champion/Challenger comparison is not possible."""

def align_features_for_evaluation(
    model,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Align evaluation features to the schema expected by a model."""
    booster = model.get_booster()
    expected_features = booster.feature_names or []

    if not expected_features:
        return features

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Evaluation data is missing model features: "
            f"{missing_features}"
        )

    return features[expected_features]

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

def compare_models(
    new_run_id: str,
    val_path: str | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Compare a candidate with the current Champion on the same validation data.

    Returns:
        A tuple containing whether the candidate is better and the calculated
        comparison metrics.

    Raises:
        ModelComparisonError:
            If the Champion cannot be loaded or evaluated safely.

        OSError, ValueError, KeyError:
            If validation data or the candidate cannot be evaluated.
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

    challenger_X_val = align_features_for_evaluation(
        challenger,
        X_val,
    )

    chall_preds = challenger.predict(
        challenger_X_val
    )
    chall_preds = inverse_transform_target(chall_preds, challenger_transform)

    chall_rmse = float(
        np.sqrt(mean_squared_error(y_val, chall_preds))
    )

    metrics = {
        "challenger_rmse": chall_rmse,
        "rmse_euro": chall_rmse,
    }
    # 3. Evaluate the current Champion
    try:
        champion_uri = f"models:/{MODEL_NAME}@champion"
        logger.info(f"Evaluating current Champion from Registry: {champion_uri}")
        
        latest_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champ_run_id = latest_version.run_id
        champion = mlflow.xgboost.load_model(champion_uri)
        
        run_info = client.get_run(champ_run_id)
        champion_X_val = align_features_for_evaluation(
            champion,
            X_val,
        )

        raw_champion_predictions = champion.predict(
            champion_X_val
        )

        if (
            run_info.data.tags.get("target_transformation")
            == "log1p"
            or run_info.data.params.get("target_transformation")
            == "log1p"
        ):
            champ_preds = np.expm1(
                raw_champion_predictions
            )
        else:
            champ_preds = raw_champion_predictions

        champ_rmse = float(
            np.sqrt(mean_squared_error(y_val, champ_preds))
        )

        metrics["champion_rmse"] = champ_rmse
        
        logger.info("--- Fair 'Real-Scale' Comparison ---")
        logger.info(f" -> Challenger RMSE: {chall_rmse:.4f}")
        logger.info(f" -> Champion RMSE:   {champ_rmse:.4f}")

        is_better = chall_rmse < champ_rmse
        return is_better, metrics

    except Exception as error:
        logger.exception(
            "Champion evaluation failed. "
            "Candidate promotion is blocked | "
            f"candidate_run_id={new_run_id}"
        )

        raise ModelComparisonError(
            "Champion/Challenger comparison failed. "
            "Candidate promotion was blocked."
        ) from error

def champion_exists() -> bool:
    """
    Return whether the configured model has a Champion alias.

    A missing alias is an expected bootstrap condition. Registry connection
    errors and other unexpected failures are propagated.
    """
    client = MlflowClient()

    try:
        client.get_model_version_by_alias(
            MODEL_NAME,
            "champion",
        )
        return True

    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False

        raise

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default_run_id"
    compare_models(run_id)