from typing import Any

import mlflow
import mlflow.pyfunc


def configure_mlflow(tracking_uri: str) -> None:
    """Configure the MLflow tracking server."""
    if not tracking_uri.strip():
        raise ValueError(
            "MLflow tracking URI must not be empty."
        )

    mlflow.set_tracking_uri(tracking_uri)


def build_registered_model_uri(
    *,
    model_name: str,
    alias: str,
) -> str:
    """Build an MLflow model-registry URI using an alias."""
    if not model_name.strip():
        raise ValueError(
            "MLflow model name must not be empty."
        )

    if not alias.strip():
        raise ValueError(
            "MLflow model alias must not be empty."
        )

    return f"models:/{model_name}@{alias}"


def load_pyfunc_model(model_uri: str) -> Any:
    """Load a registered model through MLflow's pyfunc interface."""
    if not model_uri.strip():
        raise ValueError(
            "MLflow model URI must not be empty."
        )

    return mlflow.pyfunc.load_model(model_uri)