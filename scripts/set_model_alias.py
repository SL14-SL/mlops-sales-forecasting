import argparse
import os

import mlflow
import requests
from mlflow.tracking import MlflowClient

from src.configs.loader import load_config


ENV_CFG = load_config()
MODEL_NAME = ENV_CFG["model"]["registry_name"]


def show_alias(alias: str) -> None:
    """Print the model version currently assigned to an alias."""
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        alias,
    )

    print(
        f"{MODEL_NAME}@{alias} -> "
        f"version {model_version.version} | "
        f"run_id={model_version.run_id}"
    )


def set_alias(
    *,
    alias: str,
    version: str,
) -> None:
    """Assign an existing model version to an alias."""
    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version,
    )

    print(
        f"Assigned {MODEL_NAME}@{alias} "
        f"to version {version}."
    )


def reload_api() -> None:
    """Reload the API after changing a registry alias."""
    api_url = ENV_CFG["api"]["url"]

    if api_url.endswith("/predict"):
        base_url = api_url.removesuffix(
            "/predict"
        )
    else:
        base_url = api_url.rstrip("/")

    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable is not set."
        )

    response = requests.post(
        f"{base_url}/admin/reload-serving-state",
        headers={
            "X-API-KEY": api_key,
        },
        timeout=60,
    )
    response.raise_for_status()

    print(
        f"API reload successful: {response.json()}"
    )


def parse_args() -> argparse.Namespace:
    """Parse model alias command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or update an MLflow model alias."
        )
    )

    parser.add_argument(
        "--alias",
        default="champion",
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Existing model version to assign. "
            "If omitted, only show the current version."
        ),
    )
    parser.add_argument(
        "--reload-api",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Inspect or update the configured model alias."""
    args = parse_args()

    tracking_uri = ENV_CFG[
        "tracking"
    ]["mlflow_tracking_uri"]

    mlflow.set_tracking_uri(
        tracking_uri
    )

    if args.version is None:
        show_alias(args.alias)
        return

    set_alias(
        alias=args.alias,
        version=args.version,
    )

    if args.reload_api:
        reload_api()

    show_alias(args.alias)


if __name__ == "__main__":
    main()