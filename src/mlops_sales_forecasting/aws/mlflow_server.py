"""Start MLflow with a durable RDS backend and direct S3 artifact access."""

import os
from urllib.parse import quote


def configure_server(*, secret: dict[str, str], environ: dict[str, str]) -> dict[str, str]:
    """Build server environment without exposing database credentials in arguments."""
    required = ("username", "password")
    if not all(isinstance(secret.get(key), str) and secret[key] for key in required):
        raise ValueError("RDS secret is missing username/password.")
    result = dict(environ)
    host = result["MLFLOW_DB_HOST"]
    if not host or any(char in host for char in "/?#@:"):
        raise ValueError("Invalid RDS hostname.")
    result["MLFLOW_BACKEND_STORE_URI"] = (
        f"postgresql+psycopg2://{quote(secret['username'], safe='')}:"
        f"{quote(secret['password'], safe='')}@{host}:5432/mlflow?sslmode=require"
    )
    return result


def main() -> None:
    import json

    import boto3

    response = boto3.client("secretsmanager").get_secret_value(
        SecretId=os.environ["MLFLOW_DB_SECRET_ARN"],
    )
    environment = configure_server(secret=json.loads(response["SecretString"]), environ=os.environ)
    artifact_root = environment["MLFLOW_S3_ARTIFACT_ROOT"]
    if not artifact_root.startswith("s3://"):
        raise ValueError("MLflow artifact root must be an S3 URI.")
    allowed_hosts = environment["MLFLOW_ALLOWED_HOSTS"]
    # Direct S3 uploads/downloads avoid API Gateway's payload size limit.
    # The MLflow SDK uses the training task role for artifact access.
    os.execvpe(
        "mlflow",
        [
            "mlflow",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
            "--workers",
            "1",
            "--no-serve-artifacts",
            "--default-artifact-root",
            artifact_root,
            "--allowed-hosts",
            allowed_hosts,
        ],
        environment,
    )


if __name__ == "__main__":
    main()
