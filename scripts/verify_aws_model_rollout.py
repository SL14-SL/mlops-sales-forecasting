"""Verify a dev API release switch and rollback using identical model weights."""

import argparse
import json
import math
import os
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import boto3
import httpx
from mlflow import MlflowClient

from mlops_sales_forecasting.aws.operations import refresh_service, rollback_model
from mlops_sales_forecasting.inference.bundle_loader import load_serving_bundle
from mlops_sales_forecasting.inference.model_loader import load_xgboost_model
from mlops_sales_forecasting.inference.releases.manifest import write_serving_manifest
from mlops_sales_forecasting.inference.releases.pointer import load_active_release_pointer
from mlops_sales_forecasting.inference.releases.repository import (
    activate_release,
    load_release_manifest,
)
from mlops_sales_forecasting.inference.releases.storage import build_release_paths, copy_uri


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_duplicate(*, models_path, manifest, release_root, client):
    """Copy and validate immutable artifacts before changing any live references."""
    load_serving_bundle(
        models_path=models_path,
        release_id=manifest.release_id,
        serving_alias="champion",
        model_loader=load_xgboost_model,
    )
    if not manifest.model.uri.startswith(release_root + "/"):
        raise ValueError("Expected a portable model contained in its serving release.")
    model_subpath = manifest.model.uri[len(release_root) + 1 :]
    old = client.get_model_version(manifest.model.name, manifest.model.version)
    model_id = getattr(old, "model_id", None)
    if old.source.startswith("models:/m-"):
        model_id = old.source.removeprefix("models:/")
    version = client.create_model_version(
        name=manifest.model.name,
        source=old.source,
        run_id=old.run_id,
        model_id=model_id,
        tags={"purpose": "dev-rollout-rollback-verification", "identical_weights": "true"},
        description="Technical rollout test: duplicate of the current champion; not new training.",
    )
    if str(version.status) != "READY":
        raise RuntimeError(f"Test model version is not ready: {version.status}")
    paths = build_release_paths(models_path=models_path, release_id=f"release-{uuid4()}")
    for relative in sorted({reference.path for reference in manifest.artifacts.values()}):
        copy_uri(f"{release_root}/{relative}", f"{paths['release_root']}/{relative}")
    duplicate = replace(
        manifest,
        release_id=paths["release_root"].rsplit("/", 1)[-1],
        created_at_utc=datetime.now(UTC).isoformat(),
        model=replace(
            manifest.model,
            version=str(version.version),
            uri=f"{paths['release_root']}/{model_subpath}",
        ),
        metadata={
            **(manifest.metadata or {}),
            "purpose": "dev-rollout-rollback-verification",
            "identical_weights": True,
            "source_release_id": manifest.release_id,
            "source_model_uri": f"models:/{manifest.model.name}/{version.version}",
        },
    )
    write_serving_manifest(paths["manifest"], duplicate)
    load_serving_bundle(
        models_path=models_path,
        release_id=duplicate.release_id,
        serving_alias="champion",
        model_loader=load_xgboost_model,
    )
    return duplicate


def verify_switch_and_rollback(*, config, client, original, duplicate, observe, evidence):
    """Restore the starting champion even if the forward verification fails."""
    models_path = config["paths"]["models"]
    name = original.model.name
    if load_active_release_pointer(models_path=models_path).release_id != original.release_id:
        raise RuntimeError("Active pointer changed during preparation; stopping before cutover.")
    if str(client.get_model_version_by_alias(name, "champion").version) != original.model.version:
        raise RuntimeError("Champion changed during preparation; stopping before cutover.")
    try:
        print(
            f"Switching API to {duplicate.release_id}, version {duplicate.model.version}",
            flush=True,
        )
        client.set_registered_model_alias(name, "champion", duplicate.model.version)
        activate_release(
            models_path=models_path,
            release_id=duplicate.release_id,
            previous_release_id=original.release_id,
            expected_task_type=original.task_type,
        )
        refresh_service(expected_release_id=duplicate.release_id)
        save(evidence / "after-switch.json", observe(duplicate))
    finally:
        current = load_active_release_pointer(models_path=models_path)
        if current.release_id == duplicate.release_id:
            if current.previous_release_id != original.release_id:
                raise RuntimeError("Rollback target changed; inspect the active pointer.")
            print(f"Rolling API back to {original.release_id}", flush=True)
            rollback_model(config=config)
            save(evidence / "after-rollback.json", observe(original))
        elif current.release_id == original.release_id:
            champion = str(client.get_model_version_by_alias(name, "champion").version)
            if champion == duplicate.model.version:
                client.set_registered_model_alias(name, "champion", original.model.version)
            elif champion != original.model.version:
                raise RuntimeError("Champion changed concurrently; inspect the registry.")
        else:
            raise RuntimeError("Pointer changed concurrently; automatic rollback was not applied.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-release", required=True)
    args = parser.parse_args()
    output = subprocess.check_output(
        ["terraform", "-chdir=infrastructure/terraform-aws", "output", "-json"]
    )
    values = {name: item["value"] for name, item in json.loads(output).items()}
    os.environ["AWS_DEFAULT_REGION"] = values["aws_region"]
    os.environ["AWS_ECS_CLUSTER"] = values["fargate_training_cluster_arn"]
    os.environ["AWS_API_SERVICE"] = values["api_service_name"]
    os.environ["AWS_API_ENDPOINT"] = values["api_endpoint"]
    secrets = boto3.client("secretsmanager", region_name=values["aws_region"])
    os.environ["MLFLOW_TRACKING_TOKEN"] = secrets.get_secret_value(
        SecretId=values["mlflow_token_secret_arn"]
    )["SecretString"]
    api_key = secrets.get_secret_value(SecretId=values["api_key_secret_arn"])["SecretString"]
    client = MlflowClient(tracking_uri=values["mlflow_tracking_uri"])
    models_path = values["serving_models_uri"]
    pointer = load_active_release_pointer(models_path=models_path)
    if pointer.release_id != args.expected_release:
        raise RuntimeError("Active release differs from the expected starting release.")
    original, root = load_release_manifest(models_path=models_path, release_id=pointer.release_id)
    if not original.model.name.endswith("-dev"):
        raise ValueError("This verification script is restricted to dev models.")
    if not (original.metadata or {}).get("registry_registered"):
        raise ValueError("Starting release must be backed by the shared registry.")
    config = {
        "paths": {"models": models_path},
        "tracking": {
            "model_name": original.model.name,
            "mlflow_tracking_uri": values["mlflow_tracking_uri"],
        },
    }
    evidence = Path("evidence/aws-model-rollback") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    evidence.mkdir(parents=True, exist_ok=False)
    payload = {
        "inputs": [
            {
                "Store": 1,
                "Date": "2015-04-28",
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            }
        ]
    }
    baseline_predictions = None

    def observe(manifest):
        response = httpx.get(f"{values['api_endpoint'].rstrip('/')}/readyz", timeout=60)
        response.raise_for_status()
        readiness = response.json()
        if readiness.get("active_release_id") != manifest.release_id:
            raise RuntimeError("API did not load the expected release.")
        alias = str(client.get_model_version_by_alias(original.model.name, "champion").version)
        current = load_active_release_pointer(models_path=models_path)
        if alias != manifest.model.version or current.release_id != manifest.release_id:
            raise RuntimeError("Registry champion, S3 pointer and API are inconsistent.")
        response = httpx.post(
            f"{values['api_endpoint'].rstrip('/')}/predict",
            json=payload,
            headers={"X-API-KEY": api_key},
            timeout=120,
        )
        response.raise_for_status()
        prediction = response.json()
        if (
            prediction.get("release_id") != manifest.release_id
            or prediction.get("status") != "success"
        ):
            raise RuntimeError("Prediction response did not identify the expected release.")
        numbers = [item["prediction"] for item in prediction["predictions"]]
        if len(numbers) != 1 or not all(math.isfinite(value) for value in numbers):
            raise RuntimeError("Expected one finite prediction.")
        if baseline_predictions is not None and any(
            not math.isclose(a, b, rel_tol=0, abs_tol=0.01)
            for a, b in zip(numbers, baseline_predictions, strict=True)
        ):
            raise RuntimeError("Identical model weights unexpectedly changed the prediction.")
        return {
            "pointer": current.to_dict(),
            "champion_version": alias,
            "readiness": readiness,
            "prediction": prediction,
        }

    before = observe(original)
    baseline_predictions = [item["prediction"] for item in before["prediction"]["predictions"]]
    save(evidence / "before.json", before)
    print("Validating and preparing an identical-weight test release...", flush=True)
    duplicate = prepare_duplicate(
        models_path=models_path, manifest=original, release_root=root, client=client
    )
    save(evidence / "prepared.json", duplicate.to_dict())
    print(f"Evidence directory: {evidence}", flush=True)
    verify_switch_and_rollback(
        config=config,
        client=client,
        original=original,
        duplicate=duplicate,
        observe=observe,
        evidence=evidence,
    )
    print("SUCCESS: new registry version and API release verified; original restored.", flush=True)
    print(f"Evidence: {evidence}", flush=True)


if __name__ == "__main__":
    main()
