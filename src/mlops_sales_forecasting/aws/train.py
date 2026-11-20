import argparse
import json
import shutil
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Any

import mlflow.xgboost
from mlflow.models import infer_signature

from ..data.raw.ingest import RossmannDataIngestor
from ..data.splits.split import RossmannDatasetSplitter
from ..pipeline.project_adapters import PersistingRossmannFeatureBuilder
from ..pipeline.runner import run_training_pipeline
from ..tracking.mlflow import (
    log_evaluation_result,
    log_training_result,
    start_training_run,
)
from ..training.evaluator import RossmannModelEvaluator
from ..training.trainer import RossmannModelTrainer
from .training_config import build_training_job_config


def _validate_directories(directories: list[Path]) -> None:
    """Require separate input, work, model, and output directories."""
    for first, second in combinations(directories, 2):
        if first == second or first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError(f"Job directories must not overlap: {first} and {second}")


def run_training_job(
    *,
    config_path: Path,
    input_directory: Path,
    work_directory: Path,
    model_directory: Path,
    output_directory: Path,
    estimators: int | None = None,
) -> dict[str, Any]:
    """Train and export one candidate without registry promotion."""
    input_directory = input_directory.resolve()
    work_directory = work_directory.resolve()
    model_directory = model_directory.resolve()
    output_directory = output_directory.resolve()

    _validate_directories(
        [
            input_directory,
            work_directory,
            model_directory,
            output_directory,
        ]
    )

    for directory in (work_directory, model_directory, output_directory):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Job directory must be empty: {directory}")

    config = build_training_job_config(
        config_path,
        input_directory=input_directory,
        work_directory=work_directory,
    )

    if estimators is not None:
        if estimators < 1:
            raise ValueError("estimators must be positive.")

        config["model"]["params"]["n_estimators"] = estimators

    for directory in (work_directory, model_directory, output_directory):
        directory.mkdir(parents=True, exist_ok=True)

    config["tracking"] = {
        "mlflow_tracking_uri": f"sqlite:///{work_directory / 'mlflow.db'}",
        "experiment_name": "forecasting-training-job",
    }

    with start_training_run(config, run_name="training-job") as run_id:
        result = run_training_pipeline(
            ingestor=RossmannDataIngestor(),
            feature_builder=PersistingRossmannFeatureBuilder(),
            splitter=RossmannDatasetSplitter(),
            trainer=RossmannModelTrainer(),
            evaluator=RossmannModelEvaluator(),
            config=config,
        )
        log_training_result(result.training)
        log_evaluation_result(result.evaluation)

    example = result.training.input_example
    signature = infer_signature(
        example,
        result.training.model.predict(example),
    )

    mlflow.xgboost.save_model(
        result.training.model,
        path=str(model_directory / "model"),
        signature=signature,
        pip_requirements=[
            f"{package}=={version(package)}"
            for package in (
                "mlflow",
                "xgboost",
                "pandas",
                "numpy",
                "scikit-learn",
                "scipy",
            )
        ],
        metadata={
            "model_type": "xgboost",
            "target_column": config["data"]["target_column"],
            "target_transformation": config["training"].get("target_transformation", "none"),
        },
    )

    result.datasets.require("store").to_parquet(
        model_directory / "store_metadata.parquet",
        index=False,
    )

    shutil.copy2(
        Path(config["paths"]["models"]) / "latest_state.json",
        model_directory / "latest_state.json",
    )
    shutil.copy2(
        Path(config["paths"]["features"]) / "known_calendar.parquet",
        model_directory / "known_calendar.parquet",
    )

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "training_metrics": dict(result.training.metrics),
        "evaluation_metrics": dict(result.evaluation.metrics),
        "evaluation_approved": result.evaluation.approved,
        "evaluation_reasons": list(result.evaluation.reasons),
        "training_rows": len(result.splits.train),
        "validation_rows": len(result.splits.validation),
        "parameters": dict(result.training.parameters),
        "promoted": False,
    }

    (output_directory / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one forecasting candidate in an isolated job."
    )
    parser.add_argument("command", nargs="?", choices=["train"], default="train")
    parser.add_argument("--config", type=Path, default=Path("configs/dev.yaml"))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/opt/ml/input/data/training"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/ml/work"))
    parser.add_argument("--model-dir", type=Path, default=Path("/opt/ml/model"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/output/data"),
    )
    parser.add_argument("--estimators", type=int, default=None)
    arguments = parser.parse_args()

    report = run_training_job(
        config_path=arguments.config,
        input_directory=arguments.input_dir,
        work_directory=arguments.work_dir,
        model_directory=arguments.model_dir,
        output_directory=arguments.output_dir,
        estimators=arguments.estimators,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
