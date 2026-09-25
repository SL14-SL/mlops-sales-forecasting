from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlops_sales_forecasting.inference.forecasting_policy import (
    finalize_forecasting_feature_frame,
    inject_forecasting_state_features,
    merge_request_with_calendar,
    merge_request_with_metadata,
    normalize_store_key,
    run_forecasting_feature_engineering,
)


def build_forecasting_inference_features(
    *,
    validated_frame: pd.DataFrame,
    store_metadata: pd.DataFrame,
    store_state: Mapping[str, Any],
    known_calendar: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the model-ready Rossmann inference frame."""
    normalized_frame = normalize_store_key(
        validated_frame,
        config,
    )

    features_frame = merge_request_with_metadata(
        normalized_frame,
        store_metadata,
        config,
    )

    features_frame = merge_request_with_calendar(
        features_frame,
        known_calendar,
        config,
    )

    processed_frame = (
        run_forecasting_feature_engineering(
            features_frame,
            config,
        )
    )

    processed_frame = (
        inject_forecasting_state_features(
            processed_frame,
            store_state,
            config,
        )
    )

    return finalize_forecasting_feature_frame(
        processed_frame,
        config,
    )