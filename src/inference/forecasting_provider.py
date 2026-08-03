from __future__ import annotations

import pandas as pd

from src.inference.contracts import InferenceBuildRequest
from src.inference.forecasting_policy import (
    finalize_forecasting_feature_frame,
    inject_forecasting_state_features,
    merge_request_with_calendar,
    merge_request_with_metadata,
    run_forecasting_feature_engineering,
)

def build_forecasting_inference_features(
    request: InferenceBuildRequest,
) -> pd.DataFrame:
    """Build the final forecasting inference frame."""
    validated_df = request.validated_df.copy()

    if "Store" not in validated_df.columns:
        raise ValueError(
            "Forecasting inference requires 'Store' column."
        )

    store_metadata = request.artifacts.require(
        "store_metadata"
    )
    store_state = request.artifacts.require(
        "store_state"
    )
    known_calendar = request.artifacts.require(
        "known_calendar"
    )

    validated_df["Store"] = (
        validated_df["Store"].astype(int)
    )

    store_id = int(
        validated_df["Store"].iloc[0]
    )

    features_df = merge_request_with_metadata(
        validated_df=validated_df,
        store_metadata=store_metadata,
        store_id=store_id,
    )

    features_df = merge_request_with_calendar(
        features_df,
        known_calendar,
    )

    processed_df = run_forecasting_feature_engineering(
        features_df
    )

    processed_df = inject_forecasting_state_features(
        processed_df=processed_df,
        store_state=store_state,
        store_id=store_id,
    )

    return finalize_forecasting_feature_frame(
        processed_df
    )