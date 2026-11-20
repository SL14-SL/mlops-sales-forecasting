from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def build_training_job_config(
    config_path: Path,
    *,
    input_directory: Path,
    work_directory: Path,
) -> dict[str, Any]:
    """Build an isolated local configuration for one training job."""
    with config_path.open("r", encoding="utf-8") as file:
        source = yaml.safe_load(file)

    if not isinstance(source, dict):
        raise ValueError("Training config must contain a YAML mapping.")

    config = deepcopy(source)
    input_directory = input_directory.resolve()
    work_directory = work_directory.resolve()

    if (
        input_directory == work_directory
        or input_directory.is_relative_to(work_directory)
        or work_directory.is_relative_to(input_directory)
    ):
        raise ValueError("Input and work directories must not overlap.")

    for filename in ("train.csv", "store.csv"):
        source_file = input_directory / filename

        if not source_file.is_file():
            raise FileNotFoundError(f"Training input is missing: {source_file}")

    directories = {
        "data_root": work_directory / "data",
        "raw_data": input_directory,
        "validated_data": work_directory / "data/validation",
        "interim": work_directory / "data/interim",
        "processed": work_directory / "data/processed",
        "artifacts": work_directory / "artifacts",
        "models": work_directory / "models",
        "features": work_directory / "data/features",
        "splits": work_directory / "data/splits",
        "versioning": work_directory / "data/versioning",
        "monitoring": work_directory / "data/monitoring",
        "predictions": work_directory / "data/predictions",
    }

    config["paths"] = {name: str(path) for name, path in directories.items()}
    config["environment"] = "dev"
    config["notifications"] = {"enabled": False}

    # This job does not connect to a Prefect server.
    config.pop("services", None)

    # The training entry point will explicitly configure job-local tracking.
    # Do not inherit a localhost URL or registry settings from dev.yaml.
    config.pop("tracking", None)
    config.pop("mlflow_tracking_uri", None)

    return config
