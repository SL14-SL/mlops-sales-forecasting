import os
import hashlib
import json
import time 
from datetime import datetime, timezone
import gcsfs
import mlflow
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

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

def build_recency_weights(
    dates: pd.Series,
    promo_values: pd.Series,
    weighting_config: dict,
) -> np.ndarray:
    """
    Assign higher sample weights to recent promotional observations.
    """
    parsed_dates = pd.to_datetime(
        dates,
        errors="raise",
    )

    if parsed_dates.isna().any():
        raise ValueError(
            "Training dates contain missing values."
        )

    parsed_promo = (
        pd.to_numeric(
            promo_values,
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
    )

    latest_training_date = parsed_dates.max()

    age_days = (
        latest_training_date - parsed_dates
    ).dt.days

    recent_30_day_promo = (
        parsed_promo
        & age_days.le(30)
    )

    recent_60_day_promo = (
        parsed_promo
        & age_days.gt(30)
        & age_days.le(60)
    )

    recent_120_day_promo = (
        parsed_promo
        & age_days.gt(60)
        & age_days.le(120)
    )

    weights = np.select(
        [
            recent_30_day_promo,
            recent_60_day_promo,
            recent_120_day_promo,
        ],
        [
            float(
                weighting_config.get(
                    "last_30_days_weight",
                    10.0,
                )
            ),
            float(
                weighting_config.get(
                    "last_60_days_weight",
                    5.0,
                )
            ),
            float(
                weighting_config.get(
                    "last_120_days_weight",
                    2.0,
                )
            ),
        ],
        default=float(
            weighting_config.get(
                "default_weight",
                1.0,
            )
        ),
    )

    return weights.astype(np.float32)

def build_final_refit_model_config(
    candidate_run_id: str,
) -> tuple[dict, int | None]:
    """
    Build a final model configuration using the candidate's best iteration.

    Early stopping is removed because the final model is trained on all
    available observations without a separate validation set.
    """
    model_cfg = deepcopy(TRAIN_CFG["model"])
    params = model_cfg.setdefault("params", {})

    if model_cfg["type"] != "xgboost":
        return model_cfg, None

    candidate_uri = f"runs:/{candidate_run_id}/model"
    candidate_model = mlflow.xgboost.load_model(candidate_uri)

    best_iteration = None

    try:
        best_iteration = int(candidate_model.best_iteration)
    except (AttributeError, TypeError, ValueError):
        logger.warning(
            "Candidate model does not expose a best iteration. "
            "Using configured n_estimators for final refit."
        )

    params.pop("early_stopping_rounds", None)

    if best_iteration is not None:
        params["n_estimators"] = best_iteration + 1

    return model_cfg, best_iteration

def train(
    train_file: str | None = None,
    val_file: str | None = None,
    *,
    is_drift_run: bool = False,
    run_role: str = "candidate",
    candidate_run_id: str | None = None,
):
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

    if run_role not in {"candidate", "final_refit"}:
        raise ValueError(
            f"Unsupported training run role: {run_role}"
        )

    if run_role == "final_refit" and not candidate_run_id:
        raise ValueError(
            "candidate_run_id is required for final refit."
        )

    validation_df = df_val.copy()

    if run_role == "final_refit":
        df_train = (
            pd.concat(
                [df_train, df_val],
                ignore_index=True,
            )
            .sort_values(["Date", "Store"])
            .drop_duplicates(
                subset=["Store", "Date"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Final refit dataset prepared | "
            "candidate_run_id=%s | rows=%s | start=%s | end=%s",
            candidate_run_id,
            len(df_train),
            df_train["Date"].min(),
            df_train["Date"].max(),
        )

    data_cfg = TRAIN_CFG["data"]

    candidate_best_iteration = None

    if run_role == "final_refit":
        model_cfg, candidate_best_iteration = (
            build_final_refit_model_config(
                candidate_run_id=candidate_run_id,
            )
        )
    else:
        model_cfg = deepcopy(TRAIN_CFG["model"])

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

    weighting_config = training_cfg.get(
        "recency_weighting",
        {},
    )

    use_recency_weighting = bool(
        is_drift_run
        and weighting_config.get(
            "enabled_for_drift",
            False,
        )
    )

    sample_weight = None

    if use_recency_weighting:
        if "Date" not in df_train.columns:
            raise ValueError(
                "Date column is required for recency weighting."
            )

        promo_column = weighting_config.get(
            "promo_column",
            "Promo",
        )

        if promo_column not in df_train.columns:
            raise ValueError(
                f"Promo column '{promo_column}' is missing."
            )

        sample_weight = build_recency_weights(
            dates=df_train["Date"],
            promo_values=df_train[promo_column],
            weighting_config=weighting_config,
        )

        weight_distribution = (
            pd.Series(sample_weight)
            .value_counts()
            .sort_index()
            .to_dict()
        )

        logger.info(
            "Recency weighting enabled | distribution=%s",
            weight_distribution,
        )
    else:
        logger.info(
            "Recency weighting disabled | drift_run=%s",
            is_drift_run,
        )

    X_train = df_train.drop(
        columns=drop_columns,
        errors="ignore",
    )
    X_train = normalize_feature_dtypes(X_train)

    y_train = transform_target(
        df_train[target_column],
        target_transform,
    )

    if run_role == "candidate":
        X_val = validation_df.drop(
            columns=drop_columns,
            errors="ignore",
        )
        X_val = normalize_feature_dtypes(X_val)

        y_val = transform_target(
            validation_df[target_column],
            target_transform,
        )
    else:
        X_val = None
        y_val = None

    
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
        mlflow.set_tag("is_drift_run", str(is_drift_run).lower())
        mlflow.set_tag("run_role", run_role)

        if candidate_run_id is not None:
            mlflow.set_tag(
                "candidate_run_id",
                candidate_run_id,
            )

        mlflow.log_param(
            "training_rows",
            len(df_train),
        )

        if candidate_best_iteration is not None:
            mlflow.log_param(
                "candidate_best_iteration",
                candidate_best_iteration,
            )
            mlflow.log_param(
                "final_n_estimators",
                candidate_best_iteration + 1,
            )

        mlflow.log_param("recency_weighting_enabled", use_recency_weighting)
        if sample_weight is not None:
            mlflow.log_param(
                "recent_30_days_weight",
                weighting_config["last_30_days_weight"],
            )
            mlflow.log_param(
                "recent_60_days_weight",
                weighting_config["last_60_days_weight"],
            )
            mlflow.log_param(
                "recent_120_days_weight",
                weighting_config["last_120_days_weight"],
            )
            mlflow.log_param(
                "default_weight",
                weighting_config["default_weight"],
            )
            mlflow.log_metric(
                "sample_weight_mean",
                float(sample_weight.mean()),
            )

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
            sample_weight=sample_weight,
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

        if run_role == "candidate":
            preds = model.predict(X_val)

            if evaluate_on_original_scale:
                preds_for_metric = inverse_transform_target(
                    preds,
                    target_transform,
                )
                actuals_for_metric = (
                    validation_df[target_column].to_numpy()
                )
            else:
                preds_for_metric = preds
                actuals_for_metric = y_val.to_numpy()

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        actuals_for_metric,
                        preds_for_metric,
                    )
                )
            )

            mlflow.log_metric("rmse", rmse)

            logger.info(
                "Candidate model trained | validation_rmse=%.4f",
                rmse,
            )
        else:
            candidate_run = mlflow.MlflowClient().get_run(
                candidate_run_id
            )
            candidate_rmse = candidate_run.data.metrics.get(
                "rmse"
            )

            if candidate_rmse is not None:
                mlflow.log_metric(
                    "candidate_validation_rmse",
                    candidate_rmse,
                )

            logger.info(
                "Final refit completed | "
                "training_rows=%s | candidate_run_id=%s",
                len(df_train),
                candidate_run_id,
            )

        input_example = (
            X_train
            .head(5)
            .copy()
        )

        output_example = model.predict(
            input_example
        )

        model_signature = infer_signature(
            model_input=input_example,
            model_output=output_example,
        )

        log_model_by_type(
            model=model,
            model_type=model_type,
            input_example=input_example,
            signature=model_signature,
            metadata={
                "target_column": target_column,
                "target_transformation": target_transform,
                "evaluate_on_original_scale": str(
                    evaluate_on_original_scale
                ),
                "model_type": model_type,
                "run_role": run_role,
                "candidate_run_id": candidate_run_id or "",
            },
        )

        logger.info(
            "Model logged to MLflow | "
            "run_id=%s | run_role=%s",
            "signature_input=%s",
            run_id,
            run_role,
            len(input_example.columns),
        )

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