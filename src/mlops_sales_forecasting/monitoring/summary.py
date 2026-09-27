from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import fsspec
import pandas as pd

from ..configs.paths import join_uri
from ..inference.model_manager import ModelManager
from ..storage.filesystem import file_exists
from .retraining_state import (
    build_retraining_state_path,
    load_retraining_state,
)


def _require_monitoring_path(
    config: Mapping[str, Any],
) -> str:
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a valid 'paths' section.")

    monitoring_path = paths.get("monitoring")

    if not isinstance(monitoring_path, str) or not monitoring_path:
        raise ValueError("Config must contain a non-empty 'paths.monitoring' value.")

    return monitoring_path


def _read_parquet_if_available(
    path: str,
) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()

    try:
        with fsspec.open(
            path,
            "rb",
        ) as file:
            return pd.read_parquet(file)
    except (
        OSError,
        TypeError,
        ValueError,
    ):
        return pd.DataFrame()


def _serialize_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            datetime,
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    if hasattr(value, "item"):
        value = value.item()

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


def summarize_performance(
    history: pd.DataFrame,
) -> dict[str, Any]:
    """Return the latest rolling performance window."""
    if history.empty:
        return {
            "available": False,
        }

    required_columns = {
        "window_end",
        "rmse",
        "mae",
        "bias",
        "n_samples",
    }

    if not required_columns.issubset(history.columns):
        return {
            "available": False,
            "reason": "invalid_history",
        }

    normalized = history.copy()
    normalized["window_end"] = pd.to_datetime(
        normalized["window_end"],
        errors="coerce",
        utc=True,
    )
    normalized = normalized.dropna(
        subset=[
            "window_end",
        ]
    )

    if normalized.empty:
        return {
            "available": False,
            "reason": "invalid_history",
        }

    latest = normalized.sort_values("window_end").iloc[-1]

    return {
        "available": True,
        "window_start": _serialize_value(latest.get("window_start")),
        "window_end": _serialize_value(latest["window_end"]),
        "rmse": _serialize_value(latest["rmse"]),
        "mae": _serialize_value(latest["mae"]),
        "bias": _serialize_value(latest["bias"]),
        "n_samples": int(latest["n_samples"]),
    }


def summarize_drift(
    history: pd.DataFrame,
) -> dict[str, Any]:
    """Return the latest feature-drift evaluation."""
    if history.empty:
        return {
            "available": False,
        }

    required_columns = {
        "timestamp",
        "feature",
        "drift_detected",
    }

    if not required_columns.issubset(history.columns):
        return {
            "available": False,
            "reason": "invalid_history",
        }

    normalized = history.copy()
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"],
        errors="coerce",
        utc=True,
    )
    normalized = normalized.dropna(
        subset=[
            "timestamp",
        ]
    )

    if normalized.empty:
        return {
            "available": False,
            "reason": "invalid_history",
        }

    latest_timestamp = normalized["timestamp"].max()
    latest = normalized.loc[normalized["timestamp"] == latest_timestamp]
    drifted = latest.loc[latest["drift_detected"].astype(bool)]

    return {
        "available": True,
        "timestamp": _serialize_value(latest_timestamp),
        "checked_features": int(len(latest)),
        "drifted_features": int(len(drifted)),
        "drifted_feature_names": (drifted["feature"].tolist()),
    }


def summarize_retraining(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the latest persisted retraining state."""
    if not state:
        return {
            "available": False,
        }

    return {
        "available": True,
        "last_decision_id": state.get("last_decision_id"),
        "last_retrained_at_utc": state.get("last_retrained_at_utc"),
        "action": state.get("action"),
        "trigger_types": list(
            state.get(
                "trigger_types",
                [],
            )
        ),
        "candidate_run_id": state.get("candidate_run_id"),
        "champion_promoted": bool(
            state.get(
                "champion_promoted",
                False,
            )
        ),
    }


def build_monitoring_summary(
    *,
    config: Mapping[str, Any],
    model_manager: ModelManager | None,
) -> dict[str, Any]:
    """Build the current operational monitoring summary."""
    monitoring_path = _require_monitoring_path(config)

    performance_history = _read_parquet_if_available(
        join_uri(
            monitoring_path,
            "performance_rolling.parquet",
        )
    )
    drift_history = _read_parquet_if_available(
        join_uri(
            monitoring_path,
            "feature_drift_history.parquet",
        )
    )
    retraining_state = load_retraining_state(build_retraining_state_path(monitoring_path))

    serving_ready = bool(model_manager is not None and model_manager.ready)

    return {
        "generated_at_utc": (datetime.now(UTC).isoformat()),
        "serving": {
            "ready": serving_ready,
            "active_release_id": (
                model_manager.active_release_id if model_manager is not None else None
            ),
            "last_reload_error": (
                model_manager.last_reload_error if model_manager is not None else None
            ),
        },
        "performance": summarize_performance(performance_history),
        "feature_drift": summarize_drift(drift_history),
        "retraining": summarize_retraining(retraining_state),
    }
