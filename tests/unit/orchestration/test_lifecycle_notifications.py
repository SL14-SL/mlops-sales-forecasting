from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.notifications.contracts import (
    LifecycleEventType,
)
from mlops_sales_forecasting.orchestration.lifecycle_notifications import (
    notify_candidate_outcome,
    notify_lifecycle_failure,
)


def config() -> dict:
    return {
        "project": {
            "slug": "example-project",
        },
        "environment": "test",
    }


def registration(
    *,
    registered: bool,
) -> MagicMock:
    result = MagicMock()
    result.registered = registered
    result.run_id = "run-123"
    result.model_name = "example-model"
    result.model_version = (
        "7" if registered else None
    )
    result.model_uri = (
        "models:/example-model/7"
        if registered
        else None
    )
    result.reason = (
        None
        if registered
        else "Accuracy below threshold."
    )
    return result


def promotion(
    *,
    promote: bool,
) -> MagicMock:
    decision = MagicMock()
    decision.promote = promote
    decision.metric_name = "accuracy"
    decision.candidate_value = 0.91
    decision.champion_value = 0.90
    decision.improvement = 0.01
    decision.reason = (
        "Candidate satisfies policy."
        if promote
        else "Candidate does not satisfy policy."
    )

    result = MagicMock()
    result.decision = decision
    result.previous_champion_version = "6"
    return result


def test_notifies_rejected_candidate() -> None:
    sink = MagicMock()
    candidate = MagicMock()
    candidate.registration = registration(
        registered=False
    )
    candidate.promotion = None

    notify_candidate_outcome(
        sink=sink,
        config=config(),
        candidate=candidate,
        serving_release=None,
    )

    event = sink.notify.call_args.args[0]

    assert event.event_type == (
        LifecycleEventType.CANDIDATE_REJECTED
    )
    assert event.run_id == "run-123"
    assert event.details["reason"] == (
        "Accuracy below threshold."
    )


def test_notifies_registered_challenger() -> None:
    sink = MagicMock()
    candidate = MagicMock()
    candidate.registration = registration(
        registered=True
    )
    candidate.promotion = promotion(
        promote=False
    )

    notify_candidate_outcome(
        sink=sink,
        config=config(),
        candidate=candidate,
        serving_release=None,
    )

    event = sink.notify.call_args.args[0]

    assert event.event_type == (
        LifecycleEventType.CHALLENGER_REGISTERED
    )
    assert event.details["model_version"] == "7"
    assert event.details["candidate_value"] == 0.91
    assert event.details["champion_value"] == 0.90


def test_notifies_promoted_champion_and_release() -> None:
    sink = MagicMock()
    candidate = MagicMock()
    candidate.registration = registration(
        registered=True
    )
    candidate.promotion = promotion(
        promote=True
    )

    serving_release = MagicMock()
    serving_release.manifest.release_id = (
        "release-7"
    )
    serving_release.release_root = (
        "artifacts/models/releases/release-7"
    )

    notify_candidate_outcome(
        sink=sink,
        config=config(),
        candidate=candidate,
        serving_release=serving_release,
    )

    assert sink.notify.call_count == 2

    champion_event = (
        sink.notify.call_args_list[0].args[0]
    )
    release_event = (
        sink.notify.call_args_list[1].args[0]
    )

    assert champion_event.event_type == (
        LifecycleEventType.CHAMPION_PROMOTED
    )
    assert (
        champion_event.details[
            "previous_champion_version"
        ]
        == "6"
    )

    assert release_event.event_type == (
        LifecycleEventType
        .SERVING_RELEASE_PUBLISHED
    )
    assert release_event.details["release_id"] == (
        "release-7"
    )


def test_promoted_candidate_requires_release() -> None:
    candidate = MagicMock()
    candidate.registration = registration(
        registered=True
    )
    candidate.promotion = promotion(
        promote=True
    )

    with pytest.raises(
        ValueError,
        match="published serving release",
    ):
        notify_candidate_outcome(
            sink=MagicMock(),
            config=config(),
            candidate=candidate,
            serving_release=None,
        )


def test_notifies_lifecycle_failure() -> None:
    sink = MagicMock()

    notify_lifecycle_failure(
        sink=sink,
        config=config(),
        run_id="run-123",
        error=RuntimeError(
            "training failed"
        ),
    )

    event = sink.notify.call_args.args[0]

    assert event.event_type == (
        LifecycleEventType.PIPELINE_FAILED
    )
    assert event.details == {
        "error_type": "RuntimeError",
        "error_message": "training failed",
    }