from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import fsspec
import pandas as pd

from src.configs.loader import (
    file_exists,
    get_path,
    join_uri,
    read_text,
)
from src.data.validation.validate import (
    validate_train,
)
from src.monitoring.config import (
    get_retraining_settings,
)
from src.monitoring.retraining_policy import (
    RetrainingSignals,
)
from src.monitoring.signal_evaluation import (
    evaluate_performance_degradation,
    evaluate_persistent_feature_drift,
)


RETRAINING_STATE_FILENAME = (
    "retraining_state.json"
)


def _utc_timestamp(
    value: datetime | pd.Timestamp | None = None,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(
        value or datetime.now(timezone.utc)
    )

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _list_files(pattern: str) -> list[str]:
    """
    List local or remote files while preserving their protocol.
    """

    filesystem, fs_pattern = (
        fsspec.core.url_to_fs(pattern)
    )

    matches = filesystem.glob(fs_pattern)

    return sorted(
        filesystem.unstrip_protocol(path)
        for path in matches
    )


def _read_parquet_if_available(
    path: str,
) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()

    with fsspec.open(path, "rb") as file:
        return pd.read_parquet(file)


def _read_ground_truth_batches(
    batch_files: list[str],
) -> tuple[int, str | None, bool, str | None]:
    """
    Validate all currently available Ground-Truth batches.

    The returned fingerprint is deterministic for the same file names
    and file contents and therefore works locally and with GCS.
    """

    if not batch_files:
        return (
            0,
            None,
            True,
            "No Ground-Truth batches available.",
        )

    fingerprint = hashlib.sha256()
    total_rows = 0

    for batch_path in batch_files:
        try:
            with fsspec.open(
                batch_path,
                "rb",
            ) as file:
                raw_content = file.read()

            # Do not include a local/GCS prefix in the identity.
            file_name = PurePosixPath(
                batch_path
            ).name

            fingerprint.update(
                file_name.encode("utf-8")
            )
            fingerprint.update(b"\0")
            fingerprint.update(raw_content)
            fingerprint.update(b"\0")

            batch_df = pd.read_csv(
                io.BytesIO(raw_content),
                parse_dates=["Date"],
                dtype={"StateHoliday": str},
            )

            validated_df = validate_train(
                batch_df
            )
            total_rows += len(validated_df)

        except Exception as error:
            return (
                total_rows,
                None,
                False,
                (
                    "Ground-Truth batch validation "
                    f"failed for {batch_path}: "
                    f"{error}"
                ),
            )

    dataset_version = (
        "batch-"
        f"{fingerprint.hexdigest()[:16]}"
    )

    return (
        total_rows,
        dataset_version,
        True,
        (
            f"Validated {len(batch_files)} "
            f"Ground-Truth batches."
        ),
    )


def _load_retraining_state(
    path: str,
) -> dict[str, Any]:
    if not file_exists(path):
        return {}

    try:
        payload = json.loads(read_text(path))

        if isinstance(payload, dict):
            return payload

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        pass

    return {}


def _cooldown_active(
    state: dict[str, Any],
    *,
    evaluated_at: pd.Timestamp,
    cooldown_hours: int,
) -> bool:
    last_retrained_at = state.get(
        "last_retrained_at_utc"
    )

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

    return elapsed < pd.Timedelta(
        hours=cooldown_hours
    )


def collect_retraining_signals(
    *,
    evaluated_at: datetime
    | pd.Timestamp
    | None = None,
) -> RetrainingSignals:
    """
    Collect normalized retraining signals from local or GCS storage.

    This function performs no training and no model promotion.
    """

    settings = get_retraining_settings()
    evaluation_time = _utc_timestamp(
        evaluated_at
    )

    raw_path = get_path("raw_data")
    monitoring_path = get_path("monitoring")

    batch_pattern = join_uri(
        raw_path,
        "new_batches",
        "ground_truth_*.csv",
    )
    batch_files = _list_files(batch_pattern)

    (
        new_training_rows,
        dataset_version,
        data_quality_ok,
        data_quality_reason,
    ) = _read_ground_truth_batches(
        batch_files
    )

    drift_path = join_uri(
        monitoring_path,
        "feature_drift_history.parquet",
    )
    drift_history = _read_parquet_if_available(
        drift_path
    )

    drift_settings = settings["drift"]
    drift_result = (
        evaluate_persistent_feature_drift(
            drift_history,
            evaluated_at=evaluation_time,
            lookback_days=drift_settings[
                "lookback_days"
            ],
            consecutive_windows=drift_settings[
                "consecutive_windows"
            ],
        )
    )

    performance_path = join_uri(
        monitoring_path,
        "performance_rolling.parquet",
    )
    performance_history = (
        _read_parquet_if_available(
            performance_path
        )
    )

    performance_settings = settings[
        "performance"
    ]
    performance_result = (
        evaluate_performance_degradation(
            performance_history,
            consecutive_windows=(
                performance_settings[
                    "consecutive_windows"
                ]
            ),
            rmse_limit=performance_settings[
                "rmse_limit"
            ],
            mae_limit=performance_settings[
                "mae_limit"
            ],
            absolute_bias_limit=(
                performance_settings[
                    "absolute_bias_limit"
                ]
            ),
        )
    )

    state_path = join_uri(
        monitoring_path,
        RETRAINING_STATE_FILENAME,
    )
    retraining_state = _load_retraining_state(
        state_path
    )

    cooldown_active = _cooldown_active(
        retraining_state,
        evaluated_at=evaluation_time,
        cooldown_hours=settings[
            "cooldown_hours"
        ],
    )

    maximum_rows = settings[
        "maximum_new_training_rows"
    ]
    budget_available = (
        new_training_rows <= maximum_rows
    )

    return RetrainingSignals(
        dataset_version=dataset_version,
        new_training_rows=new_training_rows,
        minimum_training_rows=settings[
            "minimum_new_training_rows"
        ],
        data_quality_ok=data_quality_ok,
        performance_degraded=(
            performance_result.triggered
        ),
        feature_drift_persistent=(
            drift_result.triggered
        ),
        cooldown_active=cooldown_active,
        budget_available=budget_available,
        performance_window_end=(
            performance_result.window_end
        ),
        drift_window_end=(
            drift_result.window_end
        ),
        performance_reason=(
            performance_result.reason
        ),
        drift_reason=drift_result.reason,
        data_quality_reason=(
            data_quality_reason
        ),
    )