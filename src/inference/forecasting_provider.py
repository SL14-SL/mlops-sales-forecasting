from __future__ import annotations

import pandas as pd

from src.inference.contracts import InferenceBuildRequest
from src.inference.forecasting_policy import (
    merge_request_with_metadata,
    run_forecasting_feature_engineering,
    inject_forecasting_state_features,
    finalize_forecasting_feature_frame,
)


def build_forecasting_inference_features(request: InferenceBuildRequest) -> pd.DataFrame:
    """
    Forecasting-specific implementation of inference feature building.

    Supports true batch inference with multiple unique stores in one request.
    """
    validated_df = request.validated_df.copy()
    store_metadata = request.artifacts.require("store_metadata")
    store_state = request.artifacts.require("store_state")

    if "Store" not in validated_df.columns:
        raise ValueError("Forecasting inference requires 'Store' column")

    validated_df["Store"] = validated_df["Store"].astype(int)

    features_df = merge_request_with_metadata(validated_df, store_metadata)
    processed_df = run_forecasting_feature_engineering(features_df)
    processed_df = inject_forecasting_state_features(processed_df, store_state)
    processed_df = finalize_forecasting_feature_frame(processed_df)

    return processed_df