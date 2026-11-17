import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import unquote, urlsplit

import pytest

from mlops_sales_forecasting.aws import mlflow_server
from mlops_sales_forecasting.aws.mlflow_ui_proxy import local_request_allowed
from mlops_sales_forecasting.configs.paths import get_project_root


def load_module(relative_path):
    path = get_project_root() / relative_path
    spec = importlib.util.spec_from_file_location(Path(relative_path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def authorizer():
    return load_module("infrastructure/terraform-aws/mlflow_authorizer/handler.py")


def test_server_encodes_rds_password_and_requires_database_tls():
    secret = {"username": "mlflow_admin", "password": "password@with:/?#%spaces !"}
    environment = {"MLFLOW_DB_HOST": "example.region.rds.amazonaws.com", "OTHER": "retained"}
    result = mlflow_server.configure_server(secret=secret, environ=environment)
    parsed = urlsplit(result["MLFLOW_BACKEND_STORE_URI"])
    assert unquote(parsed.password) == secret["password"]
    assert parsed.hostname == environment["MLFLOW_DB_HOST"]
    assert parsed.query == "sslmode=require"
    assert result["OTHER"] == "retained"
    assert "MLFLOW_BACKEND_STORE_URI" not in environment


def test_server_keeps_credentials_out_of_process_arguments(monkeypatch):
    import boto3

    secret = {"username": "mlflow_admin", "password": "secret-test-password"}
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": json.dumps(secret)}
    monkeypatch.setattr(boto3, "client", lambda _: client)
    monkeypatch.setenv("MLFLOW_DB_HOST", "db.example.com")
    monkeypatch.setenv("MLFLOW_DB_SECRET_ARN", "secret-arn")
    monkeypatch.setenv("MLFLOW_S3_ARTIFACT_ROOT", "s3://bucket/mlflow")
    monkeypatch.setenv("MLFLOW_ALLOWED_HOSTS", "example.com,127.0.0.1:5000")
    execute = MagicMock()
    monkeypatch.setattr(mlflow_server.os, "execvpe", execute)
    mlflow_server.main()
    command = execute.call_args.args[1]
    assert secret["password"] not in " ".join(command)
    assert "--no-serve-artifacts" in command
    assert command[command.index("--default-artifact-root") + 1] == "s3://bucket/mlflow"
    assert "sslmode=require" in execute.call_args.args[2]["MLFLOW_BACKEND_STORE_URI"]


@pytest.mark.parametrize("authorization", [None, "", "Basic invalid", "Bearer "])
def test_authorizer_rejects_missing_or_invalid_scheme_without_loading_secret(
    authorizer, monkeypatch, authorization
):
    client = MagicMock()
    monkeypatch.setattr(authorizer.boto3, "client", client)
    event = {"headers": {"Authorization": authorization}} if authorization is not None else {}
    assert authorizer.handler(event, None) == {"isAuthorized": False}
    client.assert_not_called()


def test_authorizer_authenticates_and_refreshes_rotated_secret(authorizer, monkeypatch, capsys):
    monkeypatch.setenv("TOKEN_SECRET_ARN", "project-token")
    client = MagicMock()
    client.get_secret_value.side_effect = [{"SecretString": "a" * 64}, {"SecretString": "b" * 64}]
    monkeypatch.setattr(authorizer.boto3, "client", lambda _: client)
    clock = [0.0]
    monkeypatch.setattr(authorizer.time, "monotonic", lambda: clock[0])
    assert authorizer.handler({"headers": {"authorization": "Bearer " + "a" * 64}}, None)[
        "isAuthorized"
    ]
    assert not authorizer.handler({"headers": {"authorization": "Bearer wrong"}}, None)[
        "isAuthorized"
    ]
    assert client.get_secret_value.call_count == 1
    clock[0] = 61.0
    assert not authorizer.handler({"headers": {"authorization": "Bearer " + "a" * 64}}, None)[
        "isAuthorized"
    ]
    assert authorizer.handler({"headers": {"Authorization": "Bearer " + "b" * 64}}, None)[
        "isAuthorized"
    ]
    assert "a" * 64 not in capsys.readouterr().out


def test_authorizer_secret_access_failure_denies_request(authorizer, monkeypatch):
    monkeypatch.setenv("TOKEN_SECRET_ARN", "project-token")
    client = MagicMock()
    client.get_secret_value.side_effect = PermissionError("access denied")
    monkeypatch.setattr(authorizer.boto3, "client", lambda _: client)
    assert authorizer.handler({"headers": {"authorization": "Bearer anything"}}, None) == {
        "isAuthorized": False
    }


@pytest.mark.parametrize("existing", [True, False])
def test_token_initializer_preserves_existing_token(existing):
    module = load_module("scripts/ensure_mlflow_token.py")
    client = MagicMock()
    client.describe_secret.return_value = (
        {"VersionIdsToStages": {"version": ["AWSCURRENT"]}} if existing else {}
    )
    assert module.ensure_token(client=client, secret_arn="token-arn") == (not existing)
    if existing:
        client.put_secret_value.assert_not_called()
    else:
        token = client.put_secret_value.call_args.kwargs["SecretString"]
        assert len(token) >= 64
        assert client.put_secret_value.call_args.kwargs["SecretId"] == "token-arn"


@pytest.mark.parametrize(
    ("deployment_id", "after", "success"),
    [
        ("new", ["new-task"], True),
        ("old", ["new-task"], False),
        ("new", ["old-task"], False),
        ("new", [], False),
    ],
)
def test_restart_requires_completed_new_deployment_and_replaced_task(deployment_id, after, success):
    module = load_module("scripts/restart_mlflow_aws.py")
    ecs = MagicMock()
    ecs.list_tasks.side_effect = [{"taskArns": ["old-task"]}, {"taskArns": after}]
    ecs.update_service.return_value = {
        "service": {"deployments": [{"status": "PRIMARY", "id": "new"}]}
    }
    ecs.describe_services.return_value = {
        "services": [
            {
                "deployments": [
                    {"status": "PRIMARY", "id": deployment_id, "rolloutState": "COMPLETED"}
                ]
            }
        ]
    }
    if success:
        module.restart_service(ecs=ecs, cluster="cluster", service="mlflow")
    else:
        with pytest.raises(RuntimeError, match="restart did not complete"):
            module.restart_service(ecs=ecs, cluster="cluster", service="mlflow")


@pytest.mark.parametrize(
    ("host", "origin", "site", "allowed"),
    [
        ("127.0.0.1:5001", None, None, True),
        ("localhost:5001", "http://localhost:5001", "same-origin", True),
        ("127.0.0.1:5001", "https://untrusted.example", "cross-site", False),
        ("untrusted.example:5001", None, None, False),
        ("127.0.0.1:5001", None, "cross-site", False),
    ],
)
def test_ui_proxy_rejects_remote_origins_and_dns_rebinding(host, origin, site, allowed):
    assert local_request_allowed(host=host, origin=origin, fetch_site=site, port=5001) == allowed


def test_real_mlflow_cli_consumes_database_uri_from_environment(monkeypatch):
    from mlflow.cli import server

    uri = "postgresql+psycopg2://user:password@database.example.com:5432/mlflow?sslmode=require"
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", uri)
    with server.make_context(
        "server", ["--no-serve-artifacts", "--default-artifact-root", "s3://bucket/mlflow"]
    ) as context:
        assert context.params["backend_store_uri"] == uri
        assert context.params["serve_artifacts"] is False


def test_mlflow_allows_internal_health_check_hosts_but_not_lookalike_domains():
    from mlflow.server.security_utils import is_allowed_host_header

    allowed = ["gateway.execute-api.eu-north-1.amazonaws.com", "127.0.0.1:5000", "10.42.*:5000"]
    assert is_allowed_host_header(allowed, "127.0.0.1:5000")
    assert is_allowed_host_header(allowed, "10.42.1.20:5000")
    assert not is_allowed_host_header(allowed, "10.42.attacker.example:5000")
