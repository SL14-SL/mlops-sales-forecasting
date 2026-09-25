from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlops_sales_forecasting.data.features.build_features import (
    preprocess_data,
)
from mlops_sales_forecasting.data.features.calendar import (
    merge_known_calendar_features,
)
from mlops_sales_forecasting.data.features.core import (
    get_lag_feature_names,
)


def _core_columns(
    config: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Return entity, target and date columns from configuration."""
    data_config = config.get("data")

    if not isinstance(data_config, Mapping):
        raise ValueError(
            "Config must contain a valid 'data' section."
        )

    id_columns = data_config.get("id_columns")

    if (
        not isinstance(id_columns, list)
        or not id_columns
        or not isinstance(id_columns[0], str)
    ):
        raise ValueError(
            "Config must define data.id_columns."
        )

    target_column = data_config.get("target_column")
    date_column = data_config.get("time_column")

    if not isinstance(target_column, str) or not target_column:
        raise ValueError(
            "Config must define data.target_column."
        )

    if not isinstance(date_column, str) or not date_column:
        raise ValueError(
            "Config must define data.time_column."
        )

    return (
        id_columns[0],
        target_column,
        date_column,
    )


def _lag_settings(
    config: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    """Return configured lag offsets and rolling windows."""
    feature_config = config.get("features", {})

    if not isinstance(feature_config, Mapping):
        raise ValueError(
            "Config must contain a valid 'features' section."
        )

    lag_config = feature_config.get(
        "lag_features",
        {},
    )

    if not isinstance(lag_config, Mapping):
        raise ValueError(
            "Config must contain valid features.lag_features."
        )

    lags = list(
        lag_config.get(
            "lags",
            [1, 7],
        )
    )
    rolling_windows = list(
        lag_config.get(
            "rolling_windows",
            [7],
        )
    )

    if not all(
        isinstance(value, int) and value > 0
        for value in [*lags, *rolling_windows]
    ):
        raise ValueError(
            "Lag offsets and rolling windows must be "
            "positive integers."
        )

    return lags, rolling_windows


def normalize_store_key(
    validated_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Normalize the configured forecasting entity key."""
    entity_column, _, _ = _core_columns(config)

    if entity_column not in validated_frame.columns:
        raise ValueError(
            "Forecasting inference requires entity column "
            f"'{entity_column}'."
        )

    normalized = validated_frame.copy()
    normalized[entity_column] = pd.to_numeric(
        normalized[entity_column],
        errors="raise",
    ).astype(int)

    return normalized


def merge_request_with_metadata(
    validated_frame: pd.DataFrame,
    store_metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Merge inference rows with static store metadata."""
    entity_column, _, _ = _core_columns(config)

    if entity_column not in store_metadata.columns:
        raise ValueError(
            "Store metadata must contain entity column "
            f"'{entity_column}'."
        )

    merged = validated_frame.merge(
        store_metadata,
        on=entity_column,
        how="left",
        validate="many_to_one",
        indicator="_metadata_merge",
    )

    missing_mask = merged["_metadata_merge"].eq(
        "left_only"
    )

    if missing_mask.any():
        missing_entities = sorted(
            merged.loc[
                missing_mask,
                entity_column,
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Store metadata is missing for "
            f"{entity_column} values: {missing_entities}."
        )

    return merged.drop(
        columns="_metadata_merge",
    )


def merge_request_with_calendar(
    features_frame: pd.DataFrame,
    known_calendar: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Attach release-bound known-calendar features."""
    entity_column, _, date_column = _core_columns(
        config
    )

    return merge_known_calendar_features(
        features_frame,
        known_calendar,
        entity_column=entity_column,
        date_column=date_column,
        strict=True,
    )


def run_forecasting_feature_engineering(
    features_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply the same configured feature pipeline as training."""
    return preprocess_data(
        features_frame,
        config=config,
        mode="inference",
    )


def inject_forecasting_state_features(
    processed_frame: pd.DataFrame,
    store_state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Replace lag placeholders with persisted target history."""
    entity_column, target_column, _ = _core_columns(
        config
    )
    lags, rolling_windows = _lag_settings(config)

    if entity_column not in processed_frame.columns:
        raise ValueError(
            "Processed inference data must contain "
            f"'{entity_column}'."
        )

    feature_names = get_lag_feature_names(
        target_column,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    result = processed_frame.copy()

    for row_index, entity_value in result[
        entity_column
    ].items():
        entity_id = int(entity_value)
        raw_history = store_state.get(
            str(entity_id),
            [],
        )

        if not isinstance(raw_history, list):
            raise ValueError(
                "Forecasting state for "
                f"{entity_column}={entity_id} must be a list."
            )

        history = [
            float(value)
            for value in raw_history
        ]

        for lag in lags:
            result.at[
                row_index,
                feature_names[f"lag_{lag}"],
            ] = (
                history[-lag]
                if len(history) >= lag
                else 0.0
            )

        for window in rolling_windows:
            result.at[
                row_index,
                feature_names[
                    f"rolling_mean_{window}"
                ],
            ] = (
                float(
                    sum(history[-window:])
                    / window
                )
                if len(history) >= window
                else 0.0
            )

    return result


def finalize_forecasting_feature_frame(
    processed_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Remove the raw date column before model inference."""
    _, _, date_column = _core_columns(config)

    return processed_frame.drop(
        columns=[date_column],
        errors="ignore",
    )


def apply_forecasting_business_rules(
    prediction: float,
    is_open: int,
) -> float:
    """Force closed-store forecasts to zero and clip negatives."""
    if is_open == 0:
        return 0.0

    return max(
        0.0,
        float(prediction),
    )