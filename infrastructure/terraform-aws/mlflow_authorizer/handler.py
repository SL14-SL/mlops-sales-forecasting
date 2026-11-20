"""HTTP API v2 authorizer for a single project-owned MLflow deployment token."""

import hmac
import os
import time

import boto3

_cached_secret = None
_cached_until = 0.0


def handler(event, context):
    del context
    global _cached_secret, _cached_until
    headers = {key.lower(): value for key, value in (event.get("headers") or {}).items()}
    authorization = headers.get("authorization", "")
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        return {"isAuthorized": False}
    token = authorization.removeprefix("Bearer ")
    if not token:
        return {"isAuthorized": False}
    try:
        now = time.monotonic()
        if _cached_secret is None or now >= _cached_until:
            secret = boto3.client("secretsmanager").get_secret_value(
                SecretId=os.environ["TOKEN_SECRET_ARN"],
            )["SecretString"]
            if not isinstance(secret, str) or len(secret) < 32:
                return {"isAuthorized": False}
            _cached_secret = secret
            _cached_until = now + 60
        authorized = hmac.compare_digest(token.encode(), _cached_secret.encode())
        return {"isAuthorized": authorized}
    except Exception:
        # Fail closed. Never log the Authorization header, token or secret value.
        return {"isAuthorized": False}
