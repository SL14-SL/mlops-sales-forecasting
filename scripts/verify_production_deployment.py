from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import requests

from src.configs.loader import get_path
from src.deployment.verification import (
    ServingVerificationError,
    verify_prediction_probe,
    verify_serving_release,
)
from src.inference.serving_release import (
    load_active_release_id,
    load_release_prediction_probe,
    load_serving_release_manifest,
)


def require_environment_variable(
    name: str,
) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable "
            f"'{name}' is not set."
        )

    return value


def verify_liveness(
    *,
    api_base_url: str,
    timeout_seconds: float = 10.0,
) -> dict:
    response = requests.get(
        f"{api_base_url.rstrip('/')}/livez",
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "alive":
        raise ServingVerificationError(
            "Production API did not report "
            "alive status."
        )

    return payload


def main() -> int:
    api_base_url = require_environment_variable(
        "PRODUCTION_API_URL"
    ).rstrip("/")

    api_key = require_environment_variable(
        "API_KEY"
    )

    models_path = get_path("models")

    expected_release_id = (
        load_active_release_id(
            models_path=models_path,
        )
    )

    if not expected_release_id:
        raise ServingVerificationError(
            "No active production serving "
            "release could be resolved."
        )

    print(
        "Verifying production deployment | "
        f"api_base_url={api_base_url} | "
        f"expected_release_id="
        f"{expected_release_id}"
    )

    liveness_payload = verify_liveness(
        api_base_url=api_base_url,
    )

    readiness_result = verify_serving_release(
        api_base_url=api_base_url,
        expected_release_id=(
            expected_release_id
        ),
        attempts=30,
        delay_seconds=2.0,
        timeout_seconds=100.0,
    )

    manifest = load_serving_release_manifest(
        models_path=models_path,
        release_id=expected_release_id,
    )

    if (
        str(readiness_result.model_version)
        != str(manifest.model_version)
    ):
        raise ServingVerificationError(
            "Production model version does not "
            "match the active release manifest | "
            f"ready="
            f"{readiness_result.model_version} | "
            f"manifest={manifest.model_version}"
        )

    if (
        readiness_result.model_run_id
        != manifest.model_run_id
    ):
        raise ServingVerificationError(
            "Production model run ID does not "
            "match the active release manifest | "
            f"ready="
            f"{readiness_result.model_run_id} | "
            f"manifest={manifest.model_run_id}"
        )

    prediction_probe_payload = (
        load_release_prediction_probe(
            models_path=models_path,
            release_id=expected_release_id,
        )
    )

    if prediction_probe_payload is None:
        raise ServingVerificationError(
            "The active production release has "
            "no semantic prediction probe."
        )

    prediction_result = (
        verify_prediction_probe(
            api_base_url=api_base_url,
            api_key=api_key,
            prediction_probe_payload=(
                prediction_probe_payload
            ),
            expected_release_id=(
                manifest.release_id
            ),
            expected_model_version=(
                manifest.model_version
            ),
            expected_model_run_id=(
                manifest.model_run_id
            ),
            attempts=3,
            delay_seconds=2.0,
            timeout_seconds=30.0,
        )
    )

    result = {
        "status": "verified",
        "api_base_url": api_base_url,
        "liveness": liveness_payload,
        "readiness": asdict(
            readiness_result
        ),
        "prediction_probe": asdict(
            prediction_result
        ),
    }

    print(
        "PRODUCTION_VERIFICATION_RESULT="
        + json.dumps(
            result,
            default=list,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            "Production deployment verification "
            f"failed: {error}",
            file=sys.stderr,
        )
        raise