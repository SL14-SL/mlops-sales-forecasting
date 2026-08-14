from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class ServingVerificationError(
    RuntimeError
):
    """Raised when a serving release cannot be verified."""


@dataclass(frozen=True)
class ServingVerificationResult:
    release_id: str
    model_version: str
    model_run_id: str
    attempts: int
    readiness_payload: dict[str, Any]


def verify_serving_release(
    *,
    api_base_url: str,
    expected_release_id: str,
    attempts: int = 20,
    delay_seconds: float = 2.0,
    timeout_seconds: float = 10.0,
) -> ServingVerificationResult:
    """
    Verify that the API serves the expected complete release.

    A successful HTTP response alone is insufficient. The active release
    identity and every required serving component must match.
    """

    readiness_url = (
        f"{api_base_url.rstrip('/')}/readyz"
    )
    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            response = requests.get(
                readiness_url,
                timeout=timeout_seconds,
            )
            response.raise_for_status()

            payload = response.json()

            if payload.get("status") != "ready":
                raise ValueError(
                    "API did not report ready status."
                )

            if (
                payload.get(
                    "serving_bundle_loaded"
                )
                is not True
            ):
                raise ValueError(
                    "No complete serving bundle "
                    "is active."
                )

            actual_release_id = payload.get(
                "release_id"
            )
            if (
                actual_release_id
                != expected_release_id
            ):
                raise ValueError(
                    "Unexpected active release: "
                    f"expected="
                    f"{expected_release_id}, "
                    f"actual="
                    f"{actual_release_id}."
                )

            required_components = {
                "store_metadata_loaded": (
                    payload.get(
                        "store_metadata_loaded"
                    )
                ),
                "state_loaded": payload.get(
                    "state_loaded"
                ),
                "calendar_loaded": payload.get(
                    "calendar_loaded"
                ),
            }

            incomplete_components = [
                name
                for name, loaded
                in required_components.items()
                if loaded is not True
            ]

            if incomplete_components:
                raise ValueError(
                    "Serving release has incomplete "
                    "components: "
                    f"{incomplete_components}."
                )

            model_version = payload.get(
                "model_version"
            )
            model_run_id = payload.get(
                "model_run_id"
            )

            if not model_version:
                raise ValueError(
                    "Ready API has no model version."
                )

            if not model_run_id:
                raise ValueError(
                    "Ready API has no model run ID."
                )

            return ServingVerificationResult(
                release_id=actual_release_id,
                model_version=str(
                    model_version
                ),
                model_run_id=str(
                    model_run_id
                ),
                attempts=attempt,
                readiness_payload=payload,
            )

        except (
            requests.RequestException,
            TypeError,
            ValueError,
        ) as error:
            last_error = error

            if attempt < attempts:
                time.sleep(delay_seconds)

    raise ServingVerificationError(
        "Serving release verification failed "
        f"after {attempts} attempts | "
        f"expected_release_id="
        f"{expected_release_id} | "
        f"last_error={last_error}"
    ) from last_error