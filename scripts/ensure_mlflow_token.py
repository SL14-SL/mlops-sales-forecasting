"""Initialize the managed MLflow token without putting its value in Terraform state."""

import json
import secrets
import subprocess

import boto3


def terraform_outputs() -> dict:
    raw = subprocess.check_output(
        [
            "terraform",
            "-chdir=infrastructure/terraform-aws",
            "output",
            "-json",
        ]
    )
    return {key: value["value"] for key, value in json.loads(raw).items()}


def ensure_token(*, client, secret_arn: str) -> bool:
    """Create the first token only; never silently rotate an existing token."""
    metadata = client.describe_secret(SecretId=secret_arn)
    if any("AWSCURRENT" in stages for stages in metadata.get("VersionIdsToStages", {}).values()):
        return False
    client.put_secret_value(SecretId=secret_arn, SecretString=secrets.token_urlsafe(48))
    return True


def main() -> None:
    outputs = terraform_outputs()
    if outputs.get("mlflow_managed_enabled") is not True:
        raise ValueError("Token initialization is only for managed MLflow.")
    created = ensure_token(
        client=boto3.client("secretsmanager"), secret_arn=outputs["mlflow_token_secret_arn"]
    )
    print("MLflow token initialized." if created else "Existing MLflow token retained.")


if __name__ == "__main__":
    main()
