from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import fsspec
import pandas as pd

from ..configs.paths import join_uri
from ..data.validation.validate import validate_train
from ..inference.releases.repository import (
    load_active_release_manifest,
)
from ..storage.filesystem import (
    file_exists,
    list_files,
)
from .retraining_policy import RetrainingSignals
from .retraining_state import (
    build_retraining_state_path,
    load_retraining_state,
)
from .signal_evaluation import (
    evaluate_performance_degradation,
    evaluate_persistent_feature_drift,
)


def _utc_timestamp(
    value: datetime | pd.Timestamp | None = None,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value or datetime.now(UTC))

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _read_parquet_if_available(
    path: str,
) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()

    with fsspec.open(path, "rb") as file:
        return pd.read_parquet(file)


def _read_ground_truth_batches(
    batch_files: list[str],
    *,
    processed_batch_ids: set[str],
) -> tuple[
    int,
    str | None,
    tuple[str, ...],
    bool,
    str | None,
]:
    """
    Validate all available Ground-Truth batches.

    Rows are counted as new only when their content-based batch ID was
    not recorded by a previous successful retraining run.
    """

    if not batch_files:
        return (
            0,
            None,
            (),
            True,
            "No Ground-Truth batches available.",
        )

    total_new_rows = 0
    current_batch_ids: set[str] = set()

    for batch_path in batch_files:
        try:
            with fsspec.open(
                batch_path,
                "rb",
            ) as file:
                raw_content = file.read()

            batch_id = "gt-" + hashlib.sha256(raw_content).hexdigest()[:20]

            batch_df = pd.read_csv(
                io.BytesIO(raw_content),
                parse_dates=["Date"],
                dtype={"StateHoliday": str},
            )

            validated_df = validate_train(batch_df)

            is_new_batch = batch_id not in processed_batch_ids and batch_id not in current_batch_ids

            if is_new_batch:
                total_new_rows += len(validated_df)

            current_batch_ids.add(batch_id)

        except Exception as error:
            return (
                total_new_rows,
                None,
                tuple(sorted(current_batch_ids)),
                False,
                (f"Ground-Truth batch validation failed for {batch_path}: {error}"),
            )

    sorted_batch_ids = tuple(sorted(current_batch_ids))

    fingerprint_payload = "|".join(sorted_batch_ids)
    dataset_version = (
        "batch-" + hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
    )

    processed_count = len(current_batch_ids & processed_batch_ids)
    new_count = len(current_batch_ids) - processed_count

    return (
        total_new_rows,
        dataset_version,
        sorted_batch_ids,
        True,
        (
            f"Validated "
            f"{len(current_batch_ids)} unique "
            "Ground-Truth batches "
            f"({new_count} new, "
            f"{processed_count} already processed)."
        ),
    )


def _cooldown_active(
    state: dict[str, Any],
    *,
    evaluated_at: pd.Timestamp,
    cooldown_hours: int,
) -> bool:
    """
    Determine whether successful retraining is still within the cooldown period.

    Returns:
        True if the cooldown period is active, otherwise False.
    """
    last_retrained_at = state.get("last_retrained_at_utc")

    if not last_retrained_at:
        return False

    parsed = pd.to_datetime(
        last_retrained_at,
        utc=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return False

    elapsed = evaluated_at - parsed

    return elapsed < pd.to_timedelta(
        cooldown_hours,
        unit="h",
    )


def _resolve_last_training_at_utc(
    state: dict[str, Any],
    *,
    models_path: str,
) -> str | None:
    """
    Resolve the latest successful training timestamp.

    Prefer the retraining state because it also represents rejected
    Candidates. Fall back to the active serving release for bootstrap.
    """

    state_timestamp = state.get("last_retrained_at_utc")

    if state_timestamp:
        return str(state_timestamp)

    try:
        manifest, _ = load_active_release_manifest(models_path=models_path)
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None

    return manifest.created_at_utc


def _scheduled_retraining_due(
    last_training_at_utc: str | None,
    *,
    evaluated_at: pd.Timestamp,
    interval_hours: int,
) -> tuple[bool, float | None]:
    """
    Determine whether the regular model refresh interval elapsed.

    Missing or invalid timestamps do not force training.
    """

    if not last_training_at_utc:
        return False, None

    parsed = pd.to_datetime(
        last_training_at_utc,
        utc=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return False, None

    elapsed = evaluated_at - parsed

    # Protect against incorrectly future-dated state.
    if elapsed < pd.Timedelta(0):
        return False, None

    elapsed_hours = elapsed.total_seconds() / 3600.0

    return (
        elapsed
        >= pd.to_timedelta(
            interval_hours,
            unit="h",
        ),
        elapsed_hours / 24.0,
    )


def _require_mapping(
    config: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = config.get(name)

    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a valid '{name}' section.")

    return value


def _require_path(
    paths: Mapping[str, Any],
    name: str,
) -> str:
    value = paths.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config must contain a non-empty 'paths.{name}' value.")

    return value


def _trigger_enabled(
    settings: Mapping[str, Any],
    name: str,
) -> bool:
    """Return whether one retraining trigger is enabled."""
    triggers = settings.get(
        "triggers",
        {},
    )

    if not isinstance(triggers, Mapping):
        raise ValueError("Retraining trigger settings must be a mapping.")

    return bool(
        triggers.get(
            name,
            True,
        )
    )


def collect_retraining_signals(
    *,
    config: Mapping[str, Any],
    evaluated_at: datetime | pd.Timestamp | None = None,
) -> RetrainingSignals:
    """Collect normalized retraining signals from storage."""
    paths = _require_mapping(
        config,
        "paths",
    )
    monitoring = _require_mapping(
        config,
        "monitoring",
    )
    settings = _require_mapping(
        monitoring,
        "retraining",
    )
    drift_settings = _require_mapping(
        settings,
        "drift",
    )
    performance_settings = _require_mapping(
        settings,
        "performance",
    )

    raw_path = _require_path(
        paths,
        "raw_data",
    )
    monitoring_path = _require_path(
        paths,
        "monitoring",
    )
    models_path = _require_path(
        paths,
        "models",
    )

    evaluation_time = _utc_timestamp(evaluated_at)
    state_path = build_retraining_state_path(monitoring_path)
    retraining_state = load_retraining_state(state_path)

    last_training_at_utc = _resolve_last_training_at_utc(
        retraining_state,
        models_path=models_path,
    )

    processed_batch_ids = set(
        retraining_state.get(
            "processed_batch_ids",
            [],
        )
    )

    batch_pattern = join_uri(
        raw_path,
        "new_batches",
        "ground_truth_*.csv",
    )
    batch_files = list_files(batch_pattern)

    (
        new_training_rows,
        dataset_version,
        batch_ids,
        data_quality_ok,
        data_quality_reason,
    ) = _read_ground_truth_batches(
        batch_files,
        processed_batch_ids=processed_batch_ids,
    )

    drift_history = _read_parquet_if_available(
        join_uri(
            monitoring_path,
            "feature_drift_history.parquet",
        )
    )
    drift_result = evaluate_persistent_feature_drift(
        drift_history,
        evaluated_at=evaluation_time,
        lookback_days=int(drift_settings["lookback_days"]),
        consecutive_windows=int(drift_settings["consecutive_windows"]),
    )

    performance_history = _read_parquet_if_available(
        join_uri(
            monitoring_path,
            "performance_rolling.parquet",
        )
    )
    performance_result = evaluate_performance_degradation(
        performance_history,
        consecutive_windows=int(performance_settings["consecutive_windows"]),
        rmse_limit=float(performance_settings["rmse_limit"]),
        mae_limit=float(performance_settings["mae_limit"]),
        absolute_bias_limit=float(performance_settings["absolute_bias_limit"]),
    )
    performance_trigger_enabled = _trigger_enabled(
        settings,
        "performance_degradation",
    )
    drift_trigger_enabled = _trigger_enabled(
        settings,
        "feature_drift",
    )
    scheduled_trigger_enabled = _trigger_enabled(
        settings,
        "scheduled_refresh",
    )
    effective_training_state = {
        **retraining_state,
        "last_retrained_at_utc": (last_training_at_utc),
    }

    cooldown_active = _cooldown_active(
        effective_training_state,
        evaluated_at=evaluation_time,
        cooldown_hours=int(settings["cooldown_hours"]),
    )

    (
        scheduled_retraining_due,
        days_since_last_training,
    ) = _scheduled_retraining_due(
        last_training_at_utc,
        evaluated_at=evaluation_time,
        interval_hours=int(settings["scheduled_interval_hours"]),
    )

    maximum_rows = int(settings["maximum_new_training_rows"])

    return RetrainingSignals(
        dataset_version=dataset_version,
        new_training_rows=new_training_rows,
        minimum_training_rows=int(settings["minimum_new_training_rows"]),
        data_quality_ok=data_quality_ok,
        performance_degraded=(performance_trigger_enabled and performance_result.triggered),
        feature_drift_persistent=(drift_trigger_enabled and drift_result.triggered),
        cooldown_active=cooldown_active,
        budget_available=(new_training_rows <= maximum_rows),
        batch_ids=batch_ids,
        scheduled_retraining_due=(scheduled_trigger_enabled and scheduled_retraining_due),
        days_since_last_training=(days_since_last_training),
        performance_window_end=(performance_result.window_end),
        drift_window_end=(drift_result.window_end),
        performance_reason=(performance_result.reason),
        drift_reason=drift_result.reason,
        data_quality_reason=(data_quality_reason),
    )
