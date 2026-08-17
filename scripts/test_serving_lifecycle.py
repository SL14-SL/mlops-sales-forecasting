from __future__ import annotations

import json
import os

import requests

from src.configs.loader import get_path
from src.deployment.verification import (
    verify_prediction_probe,
    verify_serving_release,
)
from src.inference.serving_release import (
    load_active_release_id,
    load_release_prediction_probe,
    load_serving_manifest,
)


def main() -> None:
    api_base_url = os.getenv(
        "API_BASE_URL",
        "http://localhost:8080",
    ).rstrip("/")

    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable is not set."
        )

    models_path = get_path("models")

    # 1. Verify process liveness.
    live_response = requests.get(
        f"{api_base_url}/livez",
        timeout=10,
    )
    live_response.raise_for_status()

    live_payload = live_response.json()

    if live_payload.get("status") != "alive":
        raise RuntimeError(
            "API did not report alive status."
        )

    # 2. Resolve the release expected from the
    # persisted active-release pointer.
    active_release_id = load_active_release_id(
        models_path=models_path,
    )

    if not active_release_id:
        raise RuntimeError(
            "No active serving release is configured."
        )

    manifest, _ = load_serving_manifest(
        models_path=models_path,
        release_id=active_release_id,
    )

    # 3. Verify readiness, completeness and active
    # release identity.
    readiness_result = verify_serving_release(
        api_base_url=api_base_url,
        expected_release_id=active_release_id,
        attempts=5,
        delay_seconds=1.0,
        timeout_seconds=10.0,
    )

    # 4. Compare API lineage with the immutable
    # release manifest.
    if (
        str(readiness_result.model_version)
        != str(manifest.model_version)
    ):
        raise RuntimeError(
            "Model version mismatch between API "
            "and serving manifest | "
            f"api={readiness_result.model_version} | "
            f"manifest={manifest.model_version}"
        )

    if (
        readiness_result.model_run_id
        != manifest.model_run_id
    ):
        raise RuntimeError(
            "Model run ID mismatch between API "
            "and serving manifest | "
            f"api={readiness_result.model_run_id} | "
            f"manifest={manifest.model_run_id}"
        )

    # 5. Load the checksum-validated semantic probe
    # stored with this exact release.
    prediction_probe = (
        load_release_prediction_probe(
            models_path=models_path,
            release_id=active_release_id,
        )
    )

    if prediction_probe is None:
        raise RuntimeError(
            "The active release has no semantic "
            "prediction probe. A schema-v2 release "
            "is required for the E2E test."
        )

    # 6. Execute the authenticated semantic probe.
    probe_result = verify_prediction_probe(
        api_base_url=api_base_url,
        api_key=api_key,
        prediction_probe_payload=(
            prediction_probe
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
        delay_seconds=1.0,
        timeout_seconds=30.0,
    )

    result = {
        "status": "verified",
        "api_base_url": api_base_url,
        "release_id": probe_result.release_id,
        "model_version": (
            probe_result.model_version
        ),
        "model_run_id": (
            probe_result.model_run_id
        ),
        "readiness_attempts": (
            readiness_result.attempts
        ),
        "prediction_probe_attempts": (
            probe_result.attempts
        ),
        "predictions": list(
            probe_result.predictions
        ),
    }

    print(
        "SERVING_E2E_RESULT="
        + json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()