import mlflow.sklearn
import mlflow.xgboost


MODEL_LOADERS = {
    "xgboost": mlflow.xgboost.load_model,
    "random_forest": mlflow.sklearn.load_model,
    "linear_regression": mlflow.sklearn.load_model,
}


def load_model_by_type(model_uri: str, model_type: str):
    """Load an MLflow model using the loader that matches the configured model type."""
    if model_type not in MODEL_LOADERS:
        raise ValueError(f"Unsupported model type: {model_type}")

    return MODEL_LOADERS[model_type](model_uri)

