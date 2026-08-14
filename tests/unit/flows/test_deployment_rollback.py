import pytest

from unittest.mock import MagicMock
from flows import training_flow



def test_deploy_and_verify_release_succeeds(
    monkeypatch,
):
    refresh = MagicMock()
    verify = MagicMock(
        return_value={
            "release_id": "release-new",
        }
    )
    rollback = MagicMock()

    monkeypatch.setattr(
        training_flow,
        "task_refresh_api",
        refresh,
    )
    monkeypatch.setattr(
        training_flow,
        "task_verify_serving_release",
        verify,
    )
    monkeypatch.setattr(
        training_flow,
        "task_rollback_serving_release",
        rollback,
    )

    result = (
        training_flow.deploy_and_verify_release(
            release_id="release-new",
            previous_release_id="release-old",
        )
    )

    assert result["deployment_status"] == (
        "verified"
    )
    assert result["rolled_back"] is False

    refresh.assert_called_once_with()
    verify.assert_called_once_with(
        expected_release_id="release-new",
    )
    rollback.assert_not_called()

def test_failed_verification_rolls_back(
    monkeypatch,
):
    refresh = MagicMock()
    verify = MagicMock(
        side_effect=[
            RuntimeError(
                "new release is not ready"
            ),
            {
                "release_id": "release-old",
            },
        ]
    )
    rollback = MagicMock(
        return_value={
            "release_id": "release-old",
        }
    )

    monkeypatch.setattr(
        training_flow,
        "task_refresh_api",
        refresh,
    )
    monkeypatch.setattr(
        training_flow,
        "task_verify_serving_release",
        verify,
    )
    monkeypatch.setattr(
        training_flow,
        "task_rollback_serving_release",
        rollback,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Automatic rollback completed "
            "successfully"
        ),
    ):
        training_flow.deploy_and_verify_release(
            release_id="release-new",
            previous_release_id="release-old",
        )

    rollback.assert_called_once_with(
        previous_release_id="release-old",
    )

    assert verify.call_args_list == [
        (
            (),
            {
                "expected_release_id": (
                    "release-new"
                ),
            },
        ),
        (
            (),
            {
                "expected_release_id": (
                    "release-old"
                ),
            },
        ),
    ]

def test_failed_bootstrap_verification_cannot_rollback(
    monkeypatch,
):
    monkeypatch.setattr(
        training_flow,
        "task_refresh_api",
        MagicMock(),
    )
    monkeypatch.setattr(
        training_flow,
        "task_verify_serving_release",
        MagicMock(
            side_effect=RuntimeError(
                "not ready"
            )
        ),
    )

    rollback = MagicMock()

    monkeypatch.setattr(
        training_flow,
        "task_rollback_serving_release",
        rollback,
    )

    with pytest.raises(
        RuntimeError,
        match="no previous release",
    ):
        training_flow.deploy_and_verify_release(
            release_id="release-first",
            previous_release_id=None,
        )

    rollback.assert_not_called()

def test_failed_rollback_is_reported(
    monkeypatch,
):
    monkeypatch.setattr(
        training_flow,
        "task_refresh_api",
        MagicMock(),
    )
    monkeypatch.setattr(
        training_flow,
        "task_verify_serving_release",
        MagicMock(
            side_effect=RuntimeError(
                "new release not ready"
            )
        ),
    )
    monkeypatch.setattr(
        training_flow,
        "task_rollback_serving_release",
        MagicMock(
            side_effect=RuntimeError(
                "rollback endpoint failed"
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "automatic rollback also failed"
        ),
    ):
        training_flow.deploy_and_verify_release(
            release_id="release-new",
            previous_release_id="release-old",
        )