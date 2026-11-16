import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4


def _parse_s3_prefix(uri: str) -> tuple[str, str]:
    """Parse an S3 bucket and optional object prefix."""
    parsed = urlsplit(uri)

    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or "@" in parsed.netloc
        or ":" in parsed.netloc
    ):
        raise ValueError(f"Expected an S3 prefix: {uri}")

    return parsed.netloc, parsed.path.strip("/")


def _key(prefix: str, suffix: str) -> str:
    """Join an S3 prefix and relative object key."""
    return f"{prefix}/{suffix}" if prefix else suffix


def _s3_client() -> Any:
    """Use local AWS credentials or the ECS task role."""
    import boto3

    return boto3.client("s3")


def run_fargate_training(
    *,
    input_uri: str,
    output_uri: str,
    config_path: Path,
    estimators: int | None = None,
) -> str:
    """Run one isolated training job and publish completed results to S3."""
    input_bucket, input_prefix = _parse_s3_prefix(input_uri)
    output_bucket, output_prefix = _parse_s3_prefix(output_uri)
    config_path = config_path.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    if estimators is not None and estimators < 1:
        raise ValueError("estimators must be positive.")

    run_id = f"fargate-{uuid4().hex}"
    run_prefix = _key(output_prefix, run_id)
    client = _s3_client()

    with TemporaryDirectory(prefix="forecasting-training-") as temporary:
        root = Path(temporary)
        input_directory = root / "input"
        model_directory = root / "model"
        output_directory = root / "output"
        input_directory.mkdir()

        for filename in ("train.csv", "store.csv"):
            client.download_file(
                input_bucket,
                _key(input_prefix, filename),
                str(input_directory / filename),
            )

        command = [
            sys.executable,
            "-m",
            "mlops_sales_forecasting.aws.train",
            "train",
            "--config",
            str(config_path),
            "--input-dir",
            str(input_directory),
            "--work-dir",
            str(root / "work"),
            "--model-dir",
            str(model_directory),
            "--output-dir",
            str(output_directory),
        ]

        if estimators is not None:
            command.extend(["--estimators", str(estimators)])

        # A separate process isolates MLflow state and propagates failures.
        subprocess.run(
            command,
            check=True,
            timeout=900,
            cwd=root,
        )

        required_files = [
            model_directory / "model/MLmodel",
            model_directory / "model/model.ubj",
            model_directory / "latest_state.json",
            model_directory / "known_calendar.parquet",
            output_directory / "evaluation.json",
            output_directory / "resolved_config.json",
        ]

        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(f"Training export is incomplete: {path}")

        for name, directory in (
            ("model", model_directory),
            ("output", output_directory),
        ):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(directory).as_posix()
                    client.upload_file(
                        str(path),
                        output_bucket,
                        _key(run_prefix, f"{name}/{relative}"),
                    )

        # Publish this marker only after every artifact upload succeeds.
        completion = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "input_uri": input_uri,
            "result_uri": f"s3://{output_bucket}/{run_prefix}",
            "promoted": False,
        }
        client.put_object(
            Bucket=output_bucket,
            Key=_key(run_prefix, "completion.json"),
            Body=(json.dumps(completion, indent=2) + "\n").encode("utf-8"),
            ContentType="application/json",
        )

    return completion["result_uri"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download S3 inputs, train locally, and upload candidate artifacts."
    )
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--config", type=Path, default=Path("/app/configs/dev.yaml"))
    parser.add_argument("--estimators", type=int, default=None)
    arguments = parser.parse_args()

    result_uri = run_fargate_training(
        input_uri=arguments.input_uri,
        output_uri=arguments.output_uri,
        config_path=arguments.config,
        estimators=arguments.estimators,
    )
    print(f"Training completed: {result_uri}", flush=True)


if __name__ == "__main__":
    main()
