from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
from mlflow.models import Model


@dataclass(frozen=True)
class LoadedXGBoostModel:
    """Native XGBoost model with ordered MLflow input columns."""

    estimator: Any
    input_columns: tuple[str, ...]

    def predict(
        self,
        model_input: Any,
    ) -> Any:
        """Run native XGBoost prediction."""
        return self.estimator.predict(
            model_input
        )


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
    """Build an MLflow registry URI using an alias."""
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
    """Load a model through MLflow's pyfunc interface."""
    if not model_uri.strip():
        raise ValueError(
            "MLflow model URI must not be empty."
        )

    return mlflow.pyfunc.load_model(model_uri)


def load_xgboost_model(
    model_uri: str,
) -> LoadedXGBoostModel:
    """Load native XGBoost while preserving signature columns."""
    if not model_uri.strip():
        raise ValueError(
            "MLflow model URI must not be empty."
        )

    model_metadata = Model.load(
        model_uri
    )
    signature = model_metadata.signature

    if signature is None or signature.inputs is None:
        raise ValueError(
            "MLflow XGBoost model has no input signature."
        )

    input_columns = signature.inputs.input_names()

    if (
        not input_columns
        or not all(
            isinstance(column, str) and column
            for column in input_columns
        )
    ):
        raise ValueError(
            "MLflow XGBoost model has no valid "
            "input columns."
        )

    estimator = mlflow.xgboost.load_model(
        model_uri
    )

    return LoadedXGBoostModel(
        estimator=estimator,
        input_columns=tuple(input_columns),
    )