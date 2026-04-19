import json

import fsspec
import pandas as pd

from src.configs.loader import load_config, get_path, file_exists
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def _resolve_core_columns(config: dict) -> dict:
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


def _resolve_history_length(config: dict) -> int:
    lag_cfg = config.get("features", {}).get("lag_features", {})

    lags = lag_cfg.get("lags", [1, 7])
    rolling_windows = lag_cfg.get("rolling_windows", [7])

    values = list(lags) + list(rolling_windows)

    if not values:
        return 1

    return max(values)


def create_feature_state():
    """
    Creates a snapshot of the latest known target history per entity.

    The snapshot acts as a lightweight feature state store for inference,
    so the API can reconstruct lag / rolling context in real time.

    Output shape:
        {
          "<entity_id>": [latest_target_values...]
        }
    """
    columns = _resolve_core_columns(TRAIN_CFG)
    entity_column = columns["entity_column"]
    time_column = columns["time_column"]
    target_column = columns["target_column"]
    history_length = _resolve_history_length(TRAIN_CFG)

    features_base = get_path("features")
    models_base = get_path("models")

    features_path = f"{features_base}/features.parquet"
    state_path = f"{models_base}/latest_state.json"

    if not file_exists(features_path):
        logger.error(f"No features found at {features_path}. Run build_features first.")
        return

    logger.info(f"Creating state snapshot from {features_path}")

    df = pd.read_parquet(features_path)

    required_columns = [entity_column, time_column, target_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in features data: {missing_columns}"
        )

    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])

    state_df = (
        df.sort_values([entity_column, time_column])
        .groupby(entity_column, dropna=False)
        .tail(history_length)
    )

    state_dict = (
        state_df.groupby(entity_column, dropna=False)[target_column]
        .apply(list)
        .to_dict()
    )

    state_dict = {str(entity): values for entity, values in state_dict.items()}

    try:
        with fsspec.open(state_path, "w") as f:
            json.dump(state_dict, f)

        logger.info(
            f"State snapshot saved to {state_path} "
            f"(entities={len(state_dict)}, history_length={history_length})"
        )
    except Exception as e:
        logger.error(f"Failed to save state snapshot: {str(e)}")
        raise


if __name__ == "__main__":
    create_feature_state()