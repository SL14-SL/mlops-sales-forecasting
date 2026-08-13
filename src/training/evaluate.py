import mlflow
import pandas as pd
import numpy as np
from typing import Any
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
)
from sklearn.metrics import mean_squared_error

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger
from src.training.utils import build_drop_columns
from src.training.target_transform import inverse_transform_target
from src.training.promotion_policy import evaluate_promotion_policy

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

def calculate_promotion_metrics(
    *,
    y_true: pd.Series,
    predictions: np.ndarray,
    evaluation_frame: pd.DataFrame,
) -> tuple[
    dict[str, float],
    dict[str, int],
]:
    """
    Calculate overall and business-segment metrics for promotion.

    Bias convention:
        mean(prediction - actual)

    Negative bias means underprediction.
    Positive bias means overprediction.
    """
    actual = np.asarray(
        y_true,
        dtype=float,
    )
    predicted = np.asarray(
        predictions,
        dtype=float,
    )

    if len(actual) != len(predicted):
        raise ValueError(
            "Prediction and target lengths "
            "do not match."
        )

    if "Promo" not in evaluation_frame.columns:
        raise ValueError(
            "Validation data is missing "
            "required segment column 'Promo'."
        )

    promo_values = pd.to_numeric(
        evaluation_frame["Promo"],
        errors="coerce",
    )

    if promo_values.isna().any():
        raise ValueError(
            "Validation segment column "
            "'Promo' contains invalid values."
        )

    promo_mask = (
        promo_values.to_numpy()
        == 1
    )
    non_promo_mask = (
        promo_values.to_numpy()
        == 0
    )

    def rmse_for_mask(
        mask: np.ndarray,
        segment_name: str,
    ) -> float:
        row_count = int(
            mask.sum()
        )

        if row_count == 0:
            raise ValueError(
                "No validation rows available "
                f"for segment '{segment_name}'."
            )

        return float(
            np.sqrt(
                mean_squared_error(
                    actual[mask],
                    predicted[mask],
                )
            )
        )

    metrics = {
        "overall_rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "promo_rmse": rmse_for_mask(
            promo_mask,
            "promo",
        ),
        "non_promo_rmse": rmse_for_mask(
            non_promo_mask,
            "non_promo",
        ),
        "overall_bias": float(
            np.mean(
                predicted - actual
            )
        ),
    }

    segment_rows = {
        "promo": int(
            promo_mask.sum()
        ),
        "non_promo": int(
            non_promo_mask.sum()
        ),
    }

    return metrics, segment_rows

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
) -> tuple[bool, dict[str, Any]]:
    """
    Compare a Candidate with the current Champion on the same validation data.

    The decision is made by the configured promotion policy using:

    - overall RMSE improvement
    - Promo-segment RMSE
    - Non-Promo-segment RMSE
    - overall absolute-bias regression
    - minimum validation and segment row counts

    Returns:
        A tuple containing:

        - whether the Candidate passed all promotion gates
        - calculated metrics and the complete promotion decision

    Raises:
        ModelComparisonError:
            If the Champion cannot be loaded or evaluated, the promotion
            policy cannot be evaluated, or the audit result cannot be saved.

        OSError, ValueError, KeyError:
            If validation data or the Candidate cannot be evaluated.
    """
    client = MlflowClient()

    # -------------------------------------------------
    # 1. Load shared chronological validation data
    # -------------------------------------------------
    if val_path is None:
        val_path = (
            f"{get_path('splits')}/val.parquet"
        )

    logger.info(
        "Loading validation data for model comparison: %s",
        val_path,
    )

    drop_columns = build_drop_columns(
        TRAIN_CFG
    )

    try:
        val_df = pd.read_parquet(
            val_path
        )

        target_column = TRAIN_CFG[
            "data"
        ]["target_column"]

        if target_column not in val_df.columns:
            raise KeyError(
                "Validation data is missing target "
                f"column '{target_column}'."
            )

        X_val = val_df.drop(
            columns=drop_columns,
            errors="ignore",
        )

        y_val = val_df[
            target_column
        ]

    except Exception:
        logger.exception(
            "Failed to load validation data for "
            "model comparison: %s",
            val_path,
        )
        raise

    # -------------------------------------------------
    # 2. Evaluate Candidate
    # -------------------------------------------------
    logger.info(
        "Evaluating Candidate | run_id=%s",
        new_run_id,
    )

    challenger_uri = (
        f"runs:/{new_run_id}/model"
    )

    challenger = mlflow.xgboost.load_model(
        challenger_uri
    )

    challenger_run = client.get_run(
        new_run_id
    )

    challenger_transform = (
        challenger_run.data.tags.get(
            "target_transformation"
        )
        or challenger_run.data.params.get(
            "target_transformation"
        )
        or "none"
    )

    challenger_X_val = (
        align_features_for_evaluation(
            challenger,
            X_val,
        )
    )

    raw_challenger_predictions = (
        challenger.predict(
            challenger_X_val
        )
    )

    challenger_predictions = (
        inverse_transform_target(
            raw_challenger_predictions,
            challenger_transform,
        )
    )

    candidate_metrics, segment_rows = (
        calculate_promotion_metrics(
            y_true=y_val,
            predictions=(
                challenger_predictions
            ),
            evaluation_frame=val_df,
        )
    )

    metrics: dict[str, Any] = {
        # Compatibility fields for existing logs and callers.
        "challenger_rmse": (
            candidate_metrics[
                "overall_rmse"
            ]
        ),
        "rmse_euro": (
            candidate_metrics[
                "overall_rmse"
            ]
        ),
        # Complete structured metrics.
        "candidate_metrics": (
            candidate_metrics
        ),
        "segment_rows": segment_rows,
    }

    # -------------------------------------------------
    # 3. Evaluate Champion and apply policy
    # -------------------------------------------------
    try:
        champion_uri = (
            f"models:/{MODEL_NAME}@champion"
        )

        logger.info(
            "Evaluating current Champion | "
            "model_uri=%s",
            champion_uri,
        )

        champion_version = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                "champion",
            )
        )

        champion_run_id = (
            champion_version.run_id
        )

        champion = mlflow.xgboost.load_model(
            champion_uri
        )

        champion_run = client.get_run(
            champion_run_id
        )

        champion_transform = (
            champion_run.data.tags.get(
                "target_transformation"
            )
            or champion_run.data.params.get(
                "target_transformation"
            )
            or "none"
        )

        champion_X_val = (
            align_features_for_evaluation(
                champion,
                X_val,
            )
        )

        raw_champion_predictions = (
            champion.predict(
                champion_X_val
            )
        )

        champion_predictions = (
            inverse_transform_target(
                raw_champion_predictions,
                champion_transform,
            )
        )

        champion_metrics, champion_segment_rows = (
            calculate_promotion_metrics(
                y_true=y_val,
                predictions=(
                    champion_predictions
                ),
                evaluation_frame=val_df,
            )
        )

        if champion_segment_rows != segment_rows:
            raise ModelComparisonError(
                "Champion and Candidate segment "
                "row counts do not match."
            )

        promotion_config = (
            TRAIN_CFG.get(
                "promotion",
                {},
            )
        )

        decision = evaluate_promotion_policy(
            candidate_metrics=(
                candidate_metrics
            ),
            champion_metrics=(
                champion_metrics
            ),
            validation_rows=len(
                val_df
            ),
            segment_rows=segment_rows,
            config=promotion_config,
        )

        decision_payload = {
            "policy_version": "v1",
            "candidate_run_id": (
                new_run_id
            ),
            "champion_run_id": (
                champion_run_id
            ),
            "champion_model_version": str(
                champion_version.version
            ),
            "validation_path": str(
                val_path
            ),
            "validation_rows": len(
                val_df
            ),
            "segment_rows": (
                segment_rows
            ),
            "candidate_metrics": (
                candidate_metrics
            ),
            "champion_metrics": (
                champion_metrics
            ),
            "decision": (
                decision.to_dict()
            ),
        }

        metrics.update(
            {
                "champion_rmse": (
                    champion_metrics[
                        "overall_rmse"
                    ]
                ),
                "champion_metrics": (
                    champion_metrics
                ),
                "promotion_decision": (
                    decision.to_dict()
                ),
            }
        )

        # Persist the complete decision on the Candidate run.
        # A failure here blocks automatic promotion.
        client.log_dict(
            new_run_id,
            decision_payload,
            (
                "promotion/"
                "promotion_decision.json"
            ),
        )

        client.set_tag(
            new_run_id,
            "promotion_decision",
            (
                "accepted"
                if decision.accepted
                else "rejected"
            ),
        )

        client.set_tag(
            new_run_id,
            "promotion_policy_version",
            "v1",
        )

        logger.info(
            "Promotion decision | "
            "accepted=%s | "
            "candidate_rmse=%.4f | "
            "champion_rmse=%.4f | "
            "promo_change=%.4f | "
            "non_promo_change=%.4f | "
            "reasons=%s",
            decision.accepted,
            candidate_metrics[
                "overall_rmse"
            ],
            champion_metrics[
                "overall_rmse"
            ],
            decision.checks[
                "promo_rmse"
            ].change,
            decision.checks[
                "non_promo_rmse"
            ].change,
            decision.reasons,
        )

        return (
            decision.accepted,
            metrics,
        )

    except Exception as error:
        logger.exception(
            "Champion evaluation or promotion-policy "
            "processing failed. Candidate promotion "
            "is blocked | candidate_run_id=%s",
            new_run_id,
        )

        raise ModelComparisonError(
            "Champion/Challenger comparison or "
            "promotion-policy evaluation failed. "
            "Candidate promotion was blocked."
        ) from error

    
def champion_exists() -> bool:
    """
    Return whether the configured registered model has a Champion alias.

    A missing registered model is an expected bootstrap condition.
    Unexpected registry errors are propagated.
    """
    client = MlflowClient()

    try:
        registered_model = (
            client.get_registered_model(
                MODEL_NAME
            )
        )

    except MlflowException as error:
        missing_model_codes = {
            RESOURCE_DOES_NOT_EXIST,
            ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ),
        }

        if error.error_code in missing_model_codes:
            return False

        raise

    aliases = registered_model.aliases or {}

    return "champion" in aliases


if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default_run_id"
    compare_models(run_id)