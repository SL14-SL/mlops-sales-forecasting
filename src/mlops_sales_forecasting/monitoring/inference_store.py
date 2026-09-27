from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ..configs.paths import join_uri
from ..storage.filesystem import ensure_dir


def build_inference_records(
    *,
    validated_input: pd.DataFrame,
    inference_features: pd.DataFrame,
    predictions: Sequence[float],
    release_id: str,
    request_id: str,
    feature_allowlist: Sequence[str],
    observed_at: datetime | None = None,
) -> pd.DataFrame:
    """Build privacy-bounded inference monitoring records."""
    row_count = len(validated_input)

    if row_count == 0:
        raise ValueError("Inference monitoring requires at least one row.")

    if len(inference_features) != row_count:
        raise ValueError("Inference feature count does not match validated input count.")

    if len(predictions) != row_count:
        raise ValueError("Prediction count does not match validated input count.")

    if not release_id:
        raise ValueError("Inference monitoring requires a release ID.")

    if not request_id:
        raise ValueError("Inference monitoring requires a request ID.")

    normalized_allowlist = list(dict.fromkeys(feature_allowlist))

    if not all(isinstance(feature, str) and feature for feature in normalized_allowlist):
        raise ValueError("Inference feature allowlist contains an invalid feature name.")

    input_frame = validated_input.reset_index(drop=True)
    feature_frame = inference_features.reset_index(drop=True)

    records = pd.DataFrame(index=range(row_count))

    # Identifiers required to match delayed ground truth.
    for column in (
        "Store",
        "Date",
    ):
        if column in input_frame.columns:
            records[column] = input_frame[column]
        elif column in feature_frame.columns:
            records[column] = feature_frame[column]

    missing_identifiers = {
        "Store",
        "Date",
    } - set(records.columns)

    if missing_identifiers:
        raise ValueError(
            f"Inference monitoring is missing identifiers: {sorted(missing_identifiers)}."
        )

    records["Date"] = pd.to_datetime(
        records["Date"],
        errors="coerce",
    )

    if records["Date"].isna().any():
        raise ValueError("Inference monitoring contains invalid dates.")

    for feature in normalized_allowlist:
        if feature in records.columns:
            continue

        if feature in feature_frame.columns:
            records[feature] = feature_frame[feature]
        elif feature in input_frame.columns:
            records[feature] = input_frame[feature]

    timestamp = observed_at or datetime.now(UTC)

    if timestamp.tzinfo is None:
        raise ValueError("Inference observation time must be timezone-aware.")

    records["prediction"] = [float(prediction) for prediction in predictions]
    records["release_id"] = release_id
    records["request_id"] = request_id
    records["row_index"] = range(row_count)
    records["timestamp"] = timestamp

    return records


def build_inference_log_path(
    *,
    predictions_path: str,
    observed_at: datetime,
    file_id: str,
) -> str:
    """Build one partitioned inference-log path."""
    if not predictions_path:
        raise ValueError("Predictions path must not be empty.")

    if observed_at.tzinfo is None:
        raise ValueError("Inference observation time must be timezone-aware.")

    if not file_id:
        raise ValueError("Inference log file ID must not be empty.")

    prediction_date = observed_at.astimezone(UTC).date().isoformat()

    return join_uri(
        predictions_path,
        "history",
        f"date={prediction_date}",
        f"{file_id}.parquet",
    )


def persist_inference_records(
    records: pd.DataFrame,
    *,
    predictions_path: str,
    file_id: str | None = None,
) -> str:
    """Persist one immutable inference batch."""
    if records.empty:
        raise ValueError("Cannot persist an empty inference batch.")

    if "timestamp" not in records.columns:
        raise KeyError("Inference records have no timestamp column.")

    timestamps = pd.to_datetime(
        records["timestamp"],
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        raise ValueError("Inference records contain invalid timestamps.")

    observed_at = timestamps.iloc[0].to_pydatetime()
    resolved_file_id = file_id or str(uuid4())

    output_path = build_inference_log_path(
        predictions_path=predictions_path,
        observed_at=observed_at,
        file_id=resolved_file_id,
    )

    if not output_path.startswith("gs://"):
        ensure_dir(str(Path(output_path).parent))

    records.to_parquet(
        output_path,
        index=False,
    )
    return output_path


def record_inference_batch(
    *,
    validated_input: pd.DataFrame,
    inference_features: pd.DataFrame,
    predictions: Sequence[float],
    release_id: str,
    request_id: str,
    feature_allowlist: Sequence[str],
    predictions_path: str,
    observed_at: datetime | None = None,
    file_id: str | None = None,
) -> str:
    """Build and persist one inference monitoring batch."""
    records = build_inference_records(
        validated_input=validated_input,
        inference_features=inference_features,
        predictions=predictions,
        release_id=release_id,
        request_id=request_id,
        feature_allowlist=feature_allowlist,
        observed_at=observed_at,
    )

    return persist_inference_records(
        records,
        predictions_path=predictions_path,
        file_id=file_id,
    )
