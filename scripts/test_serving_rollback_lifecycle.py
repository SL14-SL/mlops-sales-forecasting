from __future__ import annotations

import json
import os
from typing import Any

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


def _headers(api_key: str) -> dict[str, str]:
    return {
        "X-API-KEY": api_key,
    }


def _list_releases(
    *,
    api_base_url: str,
    api_key: str,
) -> dict[str, Any]:
    response = requests.get(
        (
            f"{api_base_url}"
            "/admin/serving-releases"
        ),
        headers=_headers(api_key),
        timeout=20,
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(
        payload.get("releases"),
        list,
    ):
        raise RuntimeError(
            "Serving release endpoint returned "
            "no release list."
        )

    return payload


def _activate_release(
    *,
    api_base_url: str,
    api_key: str,
    release_id: str,
) -> dict[str, Any]:
    response = requests.post(
        (
            f"{api_base_url}"
            "/admin/rollback-serving-release"
        ),
        json={
            "release_id": release_id,
        },
        headers=_headers(api_key),
        timeout=60,
    )
    response.raise_for_status()

    return response.json()


def _verify_release(
    *,
    api_base_url: str,
    api_key: str,
    models_path: str,
    release_id: str,
) -> dict[str, Any]:
    manifest, _ = load_serving_manifest(
        models_path=models_path,
        release_id=release_id,
    )

    readiness = verify_serving_release(
        api_base_url=api_base_url,
        expected_release_id=release_id,
        attempts=10,
        delay_seconds=1.0,
        timeout_seconds=10.0,
    )

    if (
        str(readiness.model_version)
        != str(manifest.model_version)
    ):
        raise RuntimeError(
            "Model version mismatch after "
            "release activation | "
            f"ready={readiness.model_version} | "
            f"manifest={manifest.model_version}"
        )

    if (
        readiness.model_run_id
        != manifest.model_run_id
    ):
        raise RuntimeError(
            "Model run ID mismatch after "
            "release activation | "
            f"ready={readiness.model_run_id} | "
            f"manifest={manifest.model_run_id}"
        )

    probe_payload = (
        load_release_prediction_probe(
            models_path=models_path,
            release_id=release_id,
        )
    )

    if probe_payload is None:
        raise RuntimeError(
            "Rollback E2E verification requires "
            "a schema-v2 release with a "
            "prediction probe | "
            f"release_id={release_id}"
        )

    probe = verify_prediction_probe(
        api_base_url=api_base_url,
        api_key=api_key,
        prediction_probe_payload=(
            probe_payload
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

    return {
        "release_id": release_id,
        "model_version": (
            probe.model_version
        ),
        "model_run_id": (
            probe.model_run_id
        ),
        "readiness_attempts": (
            readiness.attempts
        ),
        "prediction_probe_attempts": (
            probe.attempts
        ),
        "predictions": list(
            probe.predictions
        ),
    }


def _find_rollback_target(
    *,
    releases: list[dict[str, Any]],
    active_release_id: str,
    models_path: str,
) -> str:
    for release in releases:
        release_id = release.get(
            "release_id"
        )

        if (
            not release_id
            or release_id == active_release_id
        ):
            continue

        try:
            probe_payload = (
                load_release_prediction_probe(
                    models_path=models_path,
                    release_id=str(
                        release_id
                    ),
                )
            )
        except (
            FileNotFoundError,
            ValueError,
        ):
            continue

        if probe_payload is not None:
            return str(release_id)

    raise RuntimeError(
        "No inactive schema-v2 release is "
        "available for rollback testing. "
        "Publish another serving release first."
    )


def main() -> None:
    api_base_url = os.getenv(
        "API_BASE_URL",
        "http://localhost:8080",
    ).rstrip("/")

    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable "
            "is not set."
        )

    models_path = get_path("models")

    original_release_id = (
        load_active_release_id(
            models_path=models_path,
        )
    )

    if not original_release_id:
        raise RuntimeError(
            "No active serving release exists."
        )

    release_payload = _list_releases(
        api_base_url=api_base_url,
        api_key=api_key,
    )

    rollback_release_id = (
        _find_rollback_target(
            releases=release_payload[
                "releases"
            ],
            active_release_id=(
                original_release_id
            ),
            models_path=models_path,
        )
    )

    rollback_result = None
    rollback_verification = None
    restoration_result = None
    restoration_verification = None

    try:
        rollback_result = _activate_release(
            api_base_url=api_base_url,
            api_key=api_key,
            release_id=rollback_release_id,
        )

        rollback_verification = (
            _verify_release(
                api_base_url=api_base_url,
                api_key=api_key,
                models_path=models_path,
                release_id=(
                    rollback_release_id
                ),
            )
        )

    finally:
        current_release_id = (
            load_active_release_id(
                models_path=models_path,
            )
        )

        if (
            current_release_id
            != original_release_id
        ):
            restoration_result = (
                _activate_release(
                    api_base_url=api_base_url,
                    api_key=api_key,
                    release_id=(
                        original_release_id
                    ),
                )
            )

        restoration_verification = (
            _verify_release(
                api_base_url=api_base_url,
                api_key=api_key,
                models_path=models_path,
                release_id=(
                    original_release_id
                ),
            )
        )

    final_release_id = (
        load_active_release_id(
            models_path=models_path,
        )
    )

    if (
        final_release_id
        != original_release_id
    ):
        raise RuntimeError(
            "Rollback E2E test did not restore "
            "the original serving release | "
            f"expected={original_release_id} | "
            f"actual={final_release_id}"
        )

    result = {
        "status": "verified",
        "original_release_id": (
            original_release_id
        ),
        "rollback_release_id": (
            rollback_release_id
        ),
        "rollback_result": (
            rollback_result
        ),
        "rollback_verification": (
            rollback_verification
        ),
        "restoration_result": (
            restoration_result
        ),
        "restoration_verification": (
            restoration_verification
        ),
        "final_release_id": (
            final_release_id
        ),
    }

    print(
        "SERVING_ROLLBACK_E2E_RESULT="
        + json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()