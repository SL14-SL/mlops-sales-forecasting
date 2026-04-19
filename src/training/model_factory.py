import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb

from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


MODEL_REGISTRY = {
    "xgboost": xgb.XGBRegressor,
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
}

MODEL_LOGGERS = {
    "xgboost": mlflow.xgboost.log_model,
    "random_forest": mlflow.sklearn.log_model,
    "linear_regression": mlflow.sklearn.log_model,
}


def apply_repro_defaults(model_type: str, params: dict, seed: int | None) -> dict:
    resolved = deepcopy(params)

    if seed is None:
        return resolved

    if model_type == "xgboost":
        resolved.setdefault("random_state", seed)
        resolved.setdefault("seed", seed)

    if model_type == "random_forest":
        resolved.setdefault("random_state", seed)

    return resolved


def build_model(model_cfg: dict, *, seed: int | None = None):
    model_type = model_cfg["type"]
    params = model_cfg.get("params", {})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    resolved_params = apply_repro_defaults(model_type, params, seed)
    
    return MODEL_REGISTRY[model_type](**resolved_params)

def fit_model(model, model_type: str, X_train, y_train, X_val, y_val):
    if model_type == "xgboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return

    model.fit(X_train, y_train)


def log_model_by_type(
    model,
    model_type: str,
    input_example=None,
    metadata: dict | None = None,
    signature=None,
):
    if model_type not in MODEL_LOGGERS:
        raise ValueError(f"Unsupported model type for logging: {model_type}")

    kwargs = {
        "metadata": metadata or {},
    }

    if input_example is not None:
        kwargs["input_example"] = input_example

    if signature is not None:
        kwargs["signature"] = signature

    MODEL_LOGGERS[model_type](
        model,
        name="model",
        **kwargs,
    )