
import mlflow
import json
import hashlib

from datetime import datetime

from src.constants import PROJECT_ROOT
from src.configs.loader import get_path, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")

def get_training_cost_config() -> dict:
    return ENV_CFG.get("costs", {}).get("training", {})


def build_training_cost_summary(
    *,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    duration_seconds: float,
) -> dict:
    """
    Estimate runtime and infrastructure cost for one training run.
    """
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
    """
    Resolve MLflow artifact location by environment.
    """
    if ENV_CFG["environment"] == "prod":
        return get_path("models")
    return f"file://{PROJECT_ROOT / "mlruns_artifacts"}"


def get_or_create_experiment(project_name: str, artifact_location: str) -> None:
    """
    Create MLflow experiment if needed and activate it.
    """
    if not mlflow.get_experiment_by_name(project_name):
        logger.info(
            f"Creating new MLflow experiment: {project_name} at {artifact_location}"
        )
        mlflow.create_experiment(project_name, artifact_location=artifact_location)

    mlflow.set_experiment(project_name)


def build_effective_run_config() -> dict:
    """
    Build the normalized configuration that defines a reproducible training run.
    """
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
    """
    Return a deterministic hash of the effective run configuration.
    """
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def log_effective_run_config_to_mlflow(config: dict) -> None:
    """
    Persist the effective run configuration and its hash in MLflow.
    """
    mlflow.log_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False),
        "run_config/effective_config.json",
    )
