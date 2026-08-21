import pandas as pd

import mlflow

from typing import Any

from mlflow.tracking import MlflowClient

from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger

from src.training.utils import build_drop_columns
from src.training.target_transform import inverse_transform_target
from src.training.promotion_policy import evaluate_promotion_policy
from src.training.evaluate_metrics import align_features_for_evaluation, calculate_promotion_metrics

logger = get_logger(__name__)

# Load central config
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]



class ModelComparisonError(RuntimeError):
    """Raised when a safe Champion/Challenger comparison is not possible."""

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
