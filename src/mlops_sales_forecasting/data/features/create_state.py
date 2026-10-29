import json
from collections.abc import Mapping
from typing import Any

import fsspec
import pandas as pd

from mlops_sales_forecasting.data.features.dated_state import (
    build_dated_feature_state,
)
from mlops_sales_forecasting.storage.filesystem import (
    file_exists,
)
from mlops_sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_core_columns(
    config: Mapping[str, Any],
) -> dict[str, str]:
    data_cfg = config.get("data", {})

    id_columns = data_cfg.get("id_columns", [])
    if not id_columns:
        raise ValueError("Missing required config: data.id_columns")

    entity_column = id_columns[0]
    time_column = data_cfg.get("time_column")
    target_column = data_cfg.get("target_column")

    if not time_column:
        raise ValueError("Missing required config: data.time_column")

    if not target_column:
        raise ValueError("Missing required config: data.target_column")

    return {
        "entity_column": entity_column,
        "time_column": time_column,
        "target_column": target_column,
    }


def _resolve_history_length(
    config: Mapping[str, Any],
) -> int:
    lag_cfg = config.get("features", {}).get("lag_features", {})

    lags = lag_cfg.get("lags", [1, 7])
    rolling_windows = lag_cfg.get("rolling_windows", [7])

    values = list(lags) + list(rolling_windows)

    if not values:
        return 1

    return max(values)


def build_feature_state(
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build versioned, dated target history for every store."""
    columns = _resolve_core_columns(config)

    return build_dated_feature_state(
        features,
        history_length=_resolve_history_length(config),
        entity_column=columns["entity_column"],
        date_column=columns["time_column"],
        target_column=columns["target_column"],
    )

def create_feature_state(
    *,
    config: Mapping[str, Any],
    features_path: str,
    state_path: str,
) -> dict[str, Any]:
    """Create and persist the forecasting feature state."""
    if not file_exists(features_path):
        raise FileNotFoundError(f"Features not found: {features_path}")

    logger.info(
        "Creating forecasting state | source=%s",
        features_path,
    )

    features = pd.read_parquet(features_path)
    state = build_feature_state(
        features,
        config,
    )

    with fsspec.open(
        state_path,
        "w",
        auto_mkdir=True,
    ) as file:
        json.dump(
            state,
            file,
        )

    logger.info(
        "Forecasting state saved | path=%s | entities=%s",
        state_path,
        len(state["stores"]),
    )

    return state
