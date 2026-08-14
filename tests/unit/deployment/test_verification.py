from unittest.mock import MagicMock

import pytest
import requests

from src.deployment import verification


def ready_payload(
    *,
    release_id: str = "release-v2",
) -> dict:
    return {
        "status": "ready",
        "serving_bundle_loaded": True,
        "release_id": release_id,
        "model_version": "2",
        "model_run_id": "run-v2",
        "store_metadata_loaded": True,
        "state_loaded": True,
        "calendar_loaded": True,
    }


def response_with(
    payload: dict,
) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = (
        None
    )
    return response


def test_verifies_expected_release(
    monkeypatch,
):
    monkeypatch.setattr(
        verification.requests,
        "get",
        MagicMock(
            return_value=response_with(
                ready_payload()
            )
        ),
    )

    result = (
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-v2"
            ),
            attempts=1,
        )
    )

    assert result.release_id == "release-v2"
    assert result.model_version == "2"
    assert result.model_run_id == "run-v2"
    assert result.attempts == 1


def test_rejects_unexpected_release(
    monkeypatch,
):
    monkeypatch.setattr(
        verification.requests,
        "get",
        MagicMock(
            return_value=response_with(
                ready_payload(
                    release_id="release-v1"
                )
            )
        ),
    )

    with pytest.raises(
        verification.ServingVerificationError,
        match="Unexpected active release",
    ):
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-v2"
            ),
            attempts=1,
        )


def test_rejects_incomplete_bundle(
    monkeypatch,
):
    payload = ready_payload()
    payload["calendar_loaded"] = False

    monkeypatch.setattr(
        verification.requests,
        "get",
        MagicMock(
            return_value=response_with(
                payload
            )
        ),
    )

    with pytest.raises(
        verification.ServingVerificationError,
        match="incomplete components",
    ):
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-v2"
            ),
            attempts=1,
        )


def test_retries_transient_failure(
    monkeypatch,
):
    mock_get = MagicMock(
        side_effect=[
            requests.ConnectionError(
                "API restarting"
            ),
            response_with(
                ready_payload()
            ),
        ]
    )
    mock_sleep = MagicMock()

    monkeypatch.setattr(
        verification.requests,
        "get",
        mock_get,
    )
    monkeypatch.setattr(
        verification.time,
        "sleep",
        mock_sleep,
    )

    result = (
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-v2"
            ),
            attempts=2,
            delay_seconds=0.01,
        )
    )

    assert result.attempts == 2
    mock_sleep.assert_called_once_with(
        0.01
    )


def test_fails_after_all_attempts(
    monkeypatch,
):
    monkeypatch.setattr(
        verification.requests,
        "get",
        MagicMock(
            side_effect=requests.ConnectionError(
                "API unavailable"
            )
        ),
    )
    monkeypatch.setattr(
        verification.time,
        "sleep",
        MagicMock(),
    )

    with pytest.raises(
        verification.ServingVerificationError,
        match="after 2 attempts",
    ):
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-v2"
            ),
            attempts=2,
        )