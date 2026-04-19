import mlflow
from mlflow.tracking import MlflowClient

from src.configs.loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()
MODEL_NAME = CFG["model"]["registry_name"]


def register_model(run_id: str, alias: str = "champion"):
    """
    Registers a model version and assigns a registry alias.

    Supported aliases:
        - champion
        - challenger
    """
    if alias not in {"champion", "challenger"}:
        raise ValueError(f"Unsupported alias '{alias}'. Use 'champion' or 'challenger'.")

    client = MlflowClient()

    try:
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Attempting to register model from run: {run_id}")

        model_version = mlflow.register_model(model_uri, MODEL_NAME)
        version = model_version.version

        logger.info(
            f"Successfully registered version {version} of '{MODEL_NAME}'. "
            f"Assigning alias '@{alias}'."
        )

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=alias,
            version=version,
        )

        logger.info(
            f"Registry update complete: Version {version} is now tagged as '@{alias}'."
        )

        return model_version

    except Exception as e:
        logger.error(f"Failed to register model or assign alias '{alias}': {str(e)}")
        raise

if __name__ == "__main__":
    # This allows manual registration if needed
    import sys
    if len(sys.argv) > 1:
        register_model(sys.argv[1])
    else:
        # Example or hardcoded run_id for testing purposes
        test_run_id = 'dd2e7013fd7b41c9a997f5b764b3b7eb'
        logger.info(f"No Run ID provided via CLI. Using default test_run_id: {test_run_id}")
        register_model(test_run_id)