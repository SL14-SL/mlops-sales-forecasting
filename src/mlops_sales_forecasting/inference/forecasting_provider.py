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

    row_order_column = "__request_row_order"

    if row_order_column in normalized_frame.columns:
        raise ValueError("Reserved request-order column is present.")

    normalized_frame = normalized_frame.copy()
    normalized_frame[row_order_column] = range(len(normalized_frame))

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

    if row_order_column not in processed_frame.columns:
        raise ValueError("Feature pipeline lost request row identifiers.")

    processed_frame = processed_frame.sort_values(
        row_order_column,
        kind="stable",
    )

    expected_row_order = list(range(len(validated_frame)))

    if processed_frame[row_order_column].tolist() != expected_row_order:
        raise ValueError("Feature pipeline changed request row identifiers.")

    processed_frame = (
        processed_frame
        .drop(columns=row_order_column)
        .reset_index(drop=True)
    )

    return finalize_forecasting_feature_frame(
        processed_frame,
        config,
    )