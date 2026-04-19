import pandas as pd

from src.data.validation.validate import validate_inference
from src.inference.forecasting_policy import (
    normalize_store_key,
    apply_forecasting_business_rules,
)
from src.inference.forecasting_provider import build_forecasting_inference_features
from src.inference.contracts import InferenceBuildRequest


def validate_prediction_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw prediction input."""
    validated_df = validate_inference(input_df)

    # Übergangsweise noch forecasting-orientiert:
    # normalize_store_key only if present
    if "Store" in validated_df.columns:
        validated_df = normalize_store_key(validated_df)

    return validated_df


def build_inference_features(
    validated_df: pd.DataFrame,
    *,
    config: dict,
    artifacts,
    context=None,
) -> pd.DataFrame:
    """
    Build final inference feature frame using a provider-specific implementation.
    """
    project_cfg = config.get("project", {})
    problem_type = project_cfg.get("problem_type", "forecasting")

    request = InferenceBuildRequest(
        validated_df=validated_df,
        config=config,
        artifacts=artifacts,
        context=context,
    )

    if problem_type == "forecasting":
        return build_forecasting_inference_features(request)

    raise ValueError(f"Unsupported problem_type for inference: {problem_type}")


def align_features_for_model(
    processed_df: pd.DataFrame,
    model,
    model_type: str,
) -> pd.DataFrame:
    """Align inference dataframe to expected model feature order."""
    if model_type == "xgboost":
        model_features = model.get_booster().feature_names
        return processed_df[model_features]

    return processed_df


def apply_business_rules(prediction: float, is_open: int) -> float:
    """Apply project-specific post-processing rules."""
    return apply_forecasting_business_rules(prediction, is_open)


def apply_prediction_postprocessing(
    predictions: list[float],
    open_flags: list[int] | None,
) -> list[float]:
    """Apply optional domain-specific post-processing."""
    if open_flags is None:
        return predictions

    return [
        float(apply_business_rules(pred, is_open))
        for pred, is_open in zip(predictions, open_flags)
    ]