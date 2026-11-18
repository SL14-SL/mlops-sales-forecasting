"""Run the existing isolated lifecycle comparison inside an AWS task."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import boto3

from ..simulation import cli
from ..simulation.pool import count_simulation_days
from ..simulation.reporting import load_lifecycle_results, summarize_simulation_comparison
from .lifecycle import load_aws_lifecycle_config


def run_aws_simulation(
    *, maximum_days: int = 95, mode: str = "both", estimators: int | None = None
) -> dict[str, Any]:
    """Reuse the local simulator, remote MLflow and one shared comparison baseline."""
    if maximum_days < 1 or maximum_days > 95:
        raise ValueError("Simulation days must be between 1 and 95.")
    if mode not in {"both", "managed", "static"}:
        raise ValueError("Simulation mode must be both, managed or static.")
    config = deepcopy(load_aws_lifecycle_config(estimators=estimators))
    source = urlsplit(config["paths"]["raw_data"])
    if source.scheme != "s3" or not source.netloc:
        raise ValueError("AWS simulation requires an S3 raw-data path.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    prefix = f"artifacts/lifecycle_simulation/{run_id}"
    s3 = boto3.client("s3")
    # Unique names avoid changing either the live champion or another simulation.
    for key in ("experiment_name", "model_name"):
        config["tracking"][key] += f"-{run_id}"

    with TemporaryDirectory(prefix="aws-lifecycle-simulation-") as directory:
        root = Path(directory)
        raw = root / "inputs"
        raw.mkdir()
        for filename in ("train.csv", "store.csv", "simulation_ground_truth.csv"):
            s3.download_file(
                source.netloc,
                f"{source.path.strip('/')}/{filename}",
                str(raw / filename),
            )
        config["paths"]["raw_data"] = str(raw)
        available_days = count_simulation_days(
            cli.load_simulation_pool(raw / "simulation_ground_truth.csv")
        )
        if available_days < maximum_days:
            raise ValueError(
                f"Simulation source has {available_days} days; requested {maximum_days}."
            )
        config["simulation"].update(
            source_path=str(raw / "simulation_ground_truth.csv"),
            runtime_path=str(root / "runtime"),
            baseline_path=str(root / "baseline"),
            output_path=str(root / "reports"),
        )
        reports = root / "reports"
        reports.mkdir()
        modes = (
            ("disabled", "enabled")
            if mode == "both"
            else ("enabled" if mode == "managed" else "disabled",)
        )
        print(f"AWS simulation started | run_id={run_id} | mode={mode}", flush=True)
        outputs = {}
        for retraining in modes:
            name = "with_retraining" if retraining == "enabled" else "without_retraining"
            output = reports / f"{name}.csv"
            cli.main(
                [
                    "--retraining",
                    retraining,
                    "--maximum-days",
                    str(maximum_days),
                    "--output",
                    str(output),
                ],
                config=config,
            )
            frame = load_lifecycle_results(output)
            if len(frame) != maximum_days:
                raise RuntimeError(f"Simulation completed only {len(frame)}/{maximum_days} days.")
            outputs[name] = frame
            # Save each completed arm before starting another potentially long run.
            for report in sorted(reports.iterdir()):
                s3.upload_file(str(report), source.netloc, f"{prefix}/{report.name}")

        summary = {
            "run_id": run_id,
            "maximum_days": maximum_days,
            "mode": mode,
            "results_uri": f"s3://{source.netloc}/{prefix}/",
            "model_name": config["tracking"]["model_name"] + "-simulation",
        }
        if mode == "both":
            summary.update(
                summarize_simulation_comparison(
                    outputs["without_retraining"], outputs["with_retraining"]
                )
            )
        path = reports / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        s3.upload_file(str(path), source.netloc, f"{prefix}/{path.name}")
        return summary
