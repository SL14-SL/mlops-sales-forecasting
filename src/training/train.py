import os
import hashlib
import json
import time 
from datetime import datetime, timezone
import gcsfs
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.configs.loader import get_path, load_config
from src.constants import PROJECT_ROOT
from src.training.model_factory import build_model, fit_model, log_model_by_type
from src.training.target_transform import transform_target, inverse_transform_target
from src.training.utils import build_drop_columns
from src.utils.logger import get_logger


logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")

def normalize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature dtypes for model training and MLflow signature inference."""
    df = df.copy()

    object_columns = df.select_dtypes(include=["object"]).columns
    for col in object_columns:
        df[col] = df[col].astype("category")

    return df


def load_training_data(train_file: str, val_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data from local filesystem or GCS."""
    if train_file.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        df_train = pd.read_parquet(train_file, filesystem=fs)
        df_val = pd.read_parquet(val_file, filesystem=fs)
    else:
        df_train = pd.read_parquet(train_file)
        df_val = pd.read_parquet(val_file)

    return df_train, df_val

def get_training_cost_config() -> dict:
    return ENV_CFG.get("costs", {}).get("training", {})


def build_training_cost_summary(
    *,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    duration_seconds: float,
) -> dict:
    cost_cfg = get_training_cost_config()

    enabled = cost_cfg.get("enabled", False)
    hourly_rate = float(cost_cfg.get("estimated_hourly_rate", 0.0))
    currency = cost_cfg.get("currency", "EUR")

    estimated_cost = 0.0
    if enabled:
        estimated_cost = (duration_seconds / 3600.0) * hourly_rate

    return {
        "enabled": enabled,
        "currency": currency,
        "estimated_hourly_rate": hourly_rate,
        "training_started_at_utc": started_at_utc.isoformat(),
        "training_finished_at_utc": finished_at_utc.isoformat(),
        "training_duration_seconds": round(duration_seconds, 3),
        "training_duration_minutes": round(duration_seconds / 60.0, 3),
        "estimated_training_cost": round(estimated_cost, 6),
    }

def resolve_artifact_location() -> str:
    """Resolve MLflow artifact location by environment."""
    if ENV_CFG["environment"] == "prod":
        return get_path("models")
    return f"file://{PROJECT_ROOT / "mlruns_artifacts"}"


def get_or_create_experiment(project_name: str, artifact_location: str) -> None:
    """Create MLflow experiment if needed and activate it."""
    if not mlflow.get_experiment_by_name(project_name):
        logger.info(
            f"Creating new MLflow experiment: {project_name} at {artifact_location}"
        )
        mlflow.create_experiment(project_name, artifact_location=artifact_location)

    mlflow.set_experiment(project_name)


def build_effective_run_config() -> dict:
    seed = ENV_CFG.get("random_seed")

    effective_model_cfg = json.loads(json.dumps(TRAIN_CFG["model"]))
    params = effective_model_cfg.setdefault("params", {})

    if seed is not None:
        if effective_model_cfg["type"] == "xgboost":
            params.setdefault("random_state", seed)
            params.setdefault("seed", seed)
        elif effective_model_cfg["type"] == "random_forest":
            params.setdefault("random_state", seed)

    return {
        "environment_config": ENV_CFG,
        "training_config": {
            **TRAIN_CFG,
            "model": effective_model_cfg,
        },
        "repro": {
            "seed": seed,
        },
    }


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def log_effective_run_config_to_mlflow(config: dict) -> None:
    mlflow.log_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False),
        "run_config/effective_config.json",
    )


def train(train_file: str | None = None, val_file: str | None = None):
    """
    Main training task:
    - loads train/validation data
    - applies configured target transformation
    - builds model from config
    - trains and evaluates model
    - logs metadata and model to MLflow
    """
    if train_file is None or val_file is None:
        data_path = get_path("splits")
        train_file = train_file or f"{data_path}/train.parquet"
        val_file = val_file or f"{data_path}/val.parquet"

    logger.info(f"Loading training data from: {train_file}")

    try:
        df_train, df_val = load_training_data(train_file, val_file)
        logger.info(
            f"Data loaded successfully. "
            f"Train rows: {len(df_train)}, Val rows: {len(df_val)}"
        )
    except Exception as e:
        logger.error(f"Failed to load data for training: {e}")
        raise

    data_cfg = TRAIN_CFG["data"]
    model_cfg = TRAIN_CFG["model"]
    training_cfg = TRAIN_CFG.get("training", {})
    metrics_cfg = TRAIN_CFG.get("metrics", {})

    seed = ENV_CFG.get("random_seed")
    effective_cfg = build_effective_run_config()
    effective_cfg_hash = config_hash(effective_cfg)


    target_column = data_cfg["target_column"]
    target_transform = training_cfg.get("target_transformation", "none")
    evaluate_on_original_scale = metrics_cfg.get("evaluate_on_original_scale", True)
    model_type = model_cfg["type"]
    drop_columns = build_drop_columns(TRAIN_CFG)

    logger.info(
        f"Training configuration | "
        f"model_type={model_type} | "
        f"target={target_column} | "
        f"transformation={target_transform} | "
        f"drop_columns={drop_columns}"
    )

    X_train = df_train.drop(columns=drop_columns, errors="ignore")
    X_val = df_val.drop(columns=drop_columns, errors="ignore")

    X_train = normalize_feature_dtypes(X_train)
    X_val = normalize_feature_dtypes(X_val)

    y_train = transform_target(df_train[target_column], target_transform)
    y_val = transform_target(df_val[target_column], target_transform)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    project_name = ENV_CFG.get("project_name", "ml-project")
    artifact_location = resolve_artifact_location()
    get_or_create_experiment(project_name, artifact_location)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Starting model training. Run ID: {run_id}")

        mlflow.set_tag("project", project_name)
        mlflow.set_tag("env", ENV_CFG["environment"])
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("target_column", target_column)
        mlflow.set_tag("target_transformation", target_transform)

        if seed is not None:
            mlflow.log_param("seed", seed)

        mlflow.log_param("config_hash", effective_cfg_hash)
        log_effective_run_config_to_mlflow(effective_cfg)

        model = build_model(model_cfg, seed=seed)

        training_started_at_utc = datetime.now(timezone.utc)
        training_started_perf = time.perf_counter()

        fit_model(
            model=model,
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        training_finished_at_utc = datetime.now(timezone.utc)
        training_duration_seconds = time.perf_counter() - training_started_perf

        cost_summary = build_training_cost_summary(
            started_at_utc=training_started_at_utc,
            finished_at_utc=training_finished_at_utc,
            duration_seconds=training_duration_seconds,
        )

        mlflow.log_metric("training_duration_seconds", cost_summary["training_duration_seconds"])
        mlflow.log_metric("training_duration_minutes", cost_summary["training_duration_minutes"])

        if cost_summary["enabled"]:
            mlflow.log_metric("estimated_training_cost", cost_summary["estimated_training_cost"])

        mlflow.log_param("cost_currency", cost_summary["currency"])
        mlflow.log_param("estimated_hourly_rate", cost_summary["estimated_hourly_rate"])

        mlflow.log_text(
            json.dumps(cost_summary, indent=2, ensure_ascii=False),
            "cost/training_cost_summary.json",
        )

        logger.info(
            "Training cost summary | "
            f"duration_seconds={cost_summary['training_duration_seconds']} | "
            f"estimated_cost={cost_summary['estimated_training_cost']} "
            f"{cost_summary['currency']}"
        )

        mlflow.log_params(model_cfg.get("params", {}))
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("target_transformation", target_transform)
        mlflow.log_param("evaluate_on_original_scale", evaluate_on_original_scale)

        preds = model.predict(X_val)

        if evaluate_on_original_scale:
            preds_for_metric = inverse_transform_target(preds, target_transform)
            actuals_for_metric = df_val[target_column].to_numpy()
        else:
            preds_for_metric = preds
            actuals_for_metric = y_val.to_numpy()

        rmse = float(np.sqrt(mean_squared_error(actuals_for_metric, preds_for_metric)))
        mlflow.log_metric("rmse", rmse)

        log_model_by_type(
            model=model,
            model_type=model_type,
            input_example=None,
            metadata={
                "target_column": target_column,
                "target_transformation": target_transform,
                "evaluate_on_original_scale": str(evaluate_on_original_scale),
                "model_type": model_type,
            },
        )

        logger.info(f"Model logged to MLflow with RMSE: {rmse:.4f}")

        if ENV_CFG["environment"] != "prod":
            models_dir = get_path("models")
            os.makedirs(models_dir, exist_ok=True)

            if model_type == "xgboost":
                local_path = os.path.join(models_dir, "model.ubj")
                model.save_model(local_path)
                logger.info(f"Physical model file saved to: {local_path}")
            else:
                logger.info(
                    f"Skipping local native model export for model_type={model_type}. "
                    f"MLflow artifact is the primary persisted model."
                )

        return model, run_id


if __name__ == "__main__":
    train()