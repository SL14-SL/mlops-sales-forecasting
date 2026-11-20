"""Verify HTTPS auth, real registry operations and persistent S3 model artifacts."""

import argparse
import json
import os
import subprocess

import boto3
import httpx
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient


class SmokeModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return [1.0] * len(model_input)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Only verify the previously registered smoke model after a restart.",
    )
    args = parser.parse_args()
    raw = subprocess.check_output(
        [
            "terraform",
            "-chdir=infrastructure/terraform-aws",
            "output",
            "-json",
        ]
    )
    outputs = {key: value["value"] for key, value in json.loads(raw).items()}
    endpoint = outputs["mlflow_tracking_uri"]
    secret = boto3.client("secretsmanager").get_secret_value(
        SecretId=outputs["mlflow_token_secret_arn"],
    )["SecretString"]
    for headers in ({}, {"Authorization": "Bearer invalid-smoke-token"}):
        response = httpx.post(
            f"{endpoint}/api/2.0/mlflow/experiments/search",
            json={"max_results": 1},
            headers=headers,
            timeout=30,
        )
        if response.status_code not in {401, 403}:
            raise RuntimeError("MLflow gateway did not reject an unauthenticated request.")
    os.environ["MLFLOW_TRACKING_TOKEN"] = secret
    mlflow.set_tracking_uri(endpoint)
    client = MlflowClient(tracking_uri=endpoint)
    bucket = outputs["training_bucket_name"]
    # The stable name survives ECS restarts and stays separate from forecasting champions.
    model_name = "aws-infrastructure-smoke-" + bucket
    if not args.verify_existing:
        mlflow.set_experiment("aws-infrastructure-smoke")
        with mlflow.start_run(run_name="registry-persistence-smoke"):
            mlflow.log_metric("smoke_score", 1.0)
            mlflow.log_text("durable-registry-smoke\n", "smoke.txt")
            model = mlflow.pyfunc.log_model(
                name="smoke-model",
                python_model=SmokeModel(),
                input_example=pd.DataFrame({"value": [1.0]}),
                pip_requirements=[f"mlflow=={mlflow.__version__}"],
            )
        version = mlflow.register_model(model_uri=model.model_uri, name=model_name)
        client.set_registered_model_alias(
            name=model_name, alias="verified", version=str(version.version)
        )
    version = client.get_model_version_by_alias(model_name, "verified")
    run = client.get_run(version.run_id)
    if run.data.metrics.get("smoke_score") != 1.0:
        raise RuntimeError("Registry run metrics did not persist.")
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{version.version}")
    prediction = loaded.predict(pd.DataFrame({"value": [1.0]}))
    if list(prediction) != [1.0]:
        raise RuntimeError("Persisted smoke model did not reproduce its prediction.")
    print(
        json.dumps(
            {
                "endpoint": endpoint,
                "model_name": model_name,
                "model_version": str(version.version),
                "run_id": version.run_id,
                "existing_data_verified": args.verify_existing,
            }
        )
    )


if __name__ == "__main__":
    main()
