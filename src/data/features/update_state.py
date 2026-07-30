import json
import os
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd

from src.configs.loader import file_exists, get_path, load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")


def _resolve_state_columns(config: dict) -> tuple[str, str, str]:
    """Resolve the entity, time, and target columns from training config."""
    data_cfg = config.get("data", {})
    id_columns = data_cfg.get("id_columns", [])

    if not id_columns:
        raise ValueError("Missing data.id_columns in training config.")

    entity_column = id_columns[0]
    time_column = data_cfg.get("time_column")
    target_column = data_cfg.get("target_column")

    if not time_column:
        raise ValueError("Missing data.time_column in training config.")

    if not target_column:
        raise ValueError("Missing data.target_column in training config.")

    return entity_column, time_column, target_column


def _resolve_history_length(config: dict) -> int:
    """Return the history length required by all lag and rolling features."""
    lag_cfg = config.get("features", {}).get("lag_features", {})

    lags = lag_cfg.get("lags", [1, 7])
    rolling_windows = lag_cfg.get("rolling_windows", [7])

    configured_lengths = list(lags) + list(rolling_windows)

    if not configured_lengths:
        return 1

    return max(configured_lengths)


def _load_state(state_path: str) -> dict[str, list[float]]:
    """Load the current forecasting state from local or remote storage."""
    if not file_exists(state_path):
        logger.warning(
            "Feature state does not exist yet. Starting with an empty state: %s",
            state_path,
        )
        return {}

    with fsspec.open(state_path, "r") as file:
        raw_state: dict[str, Any] = json.load(file)

    state: dict[str, list[float]] = {}

    for entity_id, history in raw_state.items():
        if not isinstance(history, list):
            continue

        state[str(entity_id)] = [float(value) for value in history]

    return state


def _write_state_atomically(
    state: dict[str, list[float]],
    state_path: str,
) -> None:
    """Write state safely using a temporary file where supported."""
    if state_path.startswith("gs://"):
        temporary_path = f"{state_path}.tmp"

        with fsspec.open(temporary_path, "w") as file:
            json.dump(state, file)

        filesystem, resolved_path = fsspec.core.url_to_fs(state_path)
        _, temporary_resolved_path = fsspec.core.url_to_fs(temporary_path)

        if filesystem.exists(resolved_path):
            filesystem.rm(resolved_path)

        filesystem.mv(temporary_resolved_path, resolved_path)
        return

    local_path = Path(state_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = local_path.with_suffix(f"{local_path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file)

    os.replace(temporary_path, local_path)


def update_feature_state_from_ground_truth(
    batch_path: str,
) -> dict[str, int | str]:
    """
    Append newly available ground truth to the forecasting state.

    The function must only be called after predictions for the batch have
    already been generated. This prevents target leakage.
    """
    entity_column, time_column, target_column = _resolve_state_columns(
        TRAIN_CFG
    )
    history_length = _resolve_history_length(TRAIN_CFG)

    models_path = get_path("models")
    state_path = f"{models_path}/latest_state.json"

    if not file_exists(batch_path):
        raise FileNotFoundError(
            f"Ground-truth batch not found: {batch_path}"
        )

    batch_df = pd.read_csv(batch_path)

    required_columns = [
        entity_column,
        time_column,
        target_column,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in batch_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Ground-truth batch is missing columns: {missing_columns}"
        )

    batch_df = batch_df[required_columns].copy()
    batch_df[time_column] = pd.to_datetime(
        batch_df[time_column],
        errors="coerce",
    )
    batch_df[entity_column] = pd.to_numeric(
        batch_df[entity_column],
        errors="coerce",
    )
    batch_df[target_column] = pd.to_numeric(
        batch_df[target_column],
        errors="coerce",
    )

    batch_df = batch_df.dropna(
        subset=[
            entity_column,
            time_column,
            target_column,
        ]
    )

    batch_df[entity_column] = batch_df[entity_column].astype(int)

    batch_df = (
        batch_df.sort_values([entity_column, time_column])
        .drop_duplicates(
            subset=[entity_column, time_column],
            keep="last",
        )
    )

    state = _load_state(state_path)

    updated_entities = 0
    appended_values = 0

    for entity_id, entity_batch in batch_df.groupby(
        entity_column,
        sort=False,
    ):
        state_key = str(int(entity_id))
        history = list(state.get(state_key, []))

        new_values = (
            entity_batch.sort_values(time_column)[target_column]
            .astype(float)
            .tolist()
        )

        if not new_values:
            continue

        history.extend(new_values)
        state[state_key] = history[-history_length:]

        updated_entities += 1
        appended_values += len(new_values)

    _write_state_atomically(state, state_path)

    result: dict[str, int | str] = {
        "state_path": state_path,
        "history_length": history_length,
        "updated_entities": updated_entities,
        "appended_values": appended_values,
        "state_entities": len(state),
    }

    logger.info(
        "Feature state updated from ground truth | "
        "batch=%s | updated_entities=%s | appended_values=%s | "
        "history_length=%s",
        batch_path,
        updated_entities,
        appended_values,
        history_length,
    )

    return result