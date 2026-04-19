from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.configs.loader import get_path, file_exists
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREDICTIONS = get_path("predictions")


def _build_daily_log_path(prediction_date: str) -> str:
    return f"{PREDICTIONS}/history/date={prediction_date}/inference_log.parquet"


def log_prediction(
    input_data,
    prediction: float,
    *,
    model_alias: str | None = None,
    model_version: str | None = None,
    model_run_id: str | None = None,
    request_id: str | None = None,
    environment: str | None = None,
) -> None:
    """
    Log prediction data to:
    1. stdout as structured JSON (useful for container/cloud logging)
    2. legacy single-file Parquet history (backward compatibility)
    3. daily partitioned Parquet history (new structure)

    Notes:
    - The legacy single-file log is kept temporarily so existing monitoring code
      continues to work.
    - The daily partitioned log is the forward-looking structure.
    """
    try:
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.to_dict(orient="records")[0]

        if not isinstance(input_data, dict):
            input_data = {"raw_input": str(input_data)}

        prediction_id = str(uuid4())
        prediction_ts = datetime.now(timezone.utc)
        prediction_timestamp = prediction_ts.isoformat()
        prediction_date = prediction_ts.date().isoformat()
        resolved_environment = environment or os.getenv("APP_ENV", "dev")

        metadata = {
            "prediction_id": prediction_id,
            "prediction_timestamp": prediction_timestamp,
            "prediction_date": prediction_date,
            "environment": resolved_environment,
            "model_alias": model_alias,
            "model_version": model_version,
            "model_run_id": model_run_id,
            "request_id": request_id,
        }

        # 1) Structured stdout log for cloud/container logging
        log_entry = {
            "input": input_data,
            "prediction": prediction,
            "service": "prediction_api",
            **metadata,
        }
        print(json.dumps(log_entry), file=sys.stdout, flush=True)

        # 2) Persistent row for parquet storage
        log_data = {
            **input_data,
            "prediction": prediction,
            **metadata,
        }
        df_new = pd.DataFrame([log_data])

        # --- Legacy single-file log (keep for backward compatibility) ---
        legacy_log_file = os.path.join(PREDICTIONS, "inference_log.parquet")

        if file_exists(legacy_log_file):
            df_existing = pd.read_parquet(legacy_log_file)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_parquet(legacy_log_file, index=False)

        # --- New daily-partitioned log ---
        daily_log_file = _build_daily_log_path(prediction_date)

        if file_exists(daily_log_file):
            daily_existing = pd.read_parquet(daily_log_file)
            daily_all = pd.concat([daily_existing, df_new], ignore_index=True)
        else:
            daily_all = df_new

        if not daily_log_file.startswith("gs://"):
            Path(daily_log_file).parent.mkdir(parents=True, exist_ok=True)

        daily_all.to_parquet(daily_log_file, index=False)

        logger.info(
            "Prediction logged successfully to %s and %s",
            legacy_log_file,
            daily_log_file,
        )

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")