from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.inference import model_manager
from mlops_sales_forecasting.inference.model_manager import (
    ModelManager,
    ModelNotReadyError,
)
from mlops_sales_forecasting.monitoring.serving import (
    SERVING_READY,
)


def bundle(release_id: str) -> MagicMock:
    result = MagicMock()
    result.release_id = release_id
    return result


def allow_mock_bundles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_manager,
        "validate_serving_bundle",
        lambda _: None,
    )


def serving_readiness() -> float:
    return SERVING_READY._value.get()
    
    
def test_manager_is_initially_not_ready() -> None:
    manager = ModelManager(MagicMock())

    assert manager.ready is False
    assert manager.active_release_id is None
    assert serving_readiness() == 0

    with pytest.raises(
        ModelNotReadyError,
        match="No serving bundle",
    ):
        manager.get_bundle()


def test_load_initial_sets_active_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_mock_bundles(monkeypatch)
    initial_bundle = bundle("release-1")
    manager = ModelManager(
        MagicMock(return_value=initial_bundle)
    )

    result = manager.load_initial()

    assert result is initial_bundle
    assert manager.ready is True
    assert manager.get_bundle() is initial_bundle
    assert manager.active_release_id == "release-1"
    assert manager.last_reload_error is None
    assert serving_readiness() == 1


def test_load_initial_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_mock_bundles(monkeypatch)
    loader = MagicMock(
        side_effect=RuntimeError("MLflow unavailable")
    )
    manager = ModelManager(loader)

    with pytest.raises(
        RuntimeError,
        match="MLflow unavailable",
    ):
        manager.load_initial()

    assert manager.ready is False
    assert manager.last_reload_error == "MLflow unavailable"
    assert serving_readiness() == 0


def test_reload_replaces_active_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_mock_bundles(monkeypatch)
    first_bundle = bundle("release-1")
    second_bundle = bundle("release-2")
    loader = MagicMock(
        side_effect=[
            first_bundle,
            second_bundle,
        ]
    )
    manager = ModelManager(loader)
    manager.load_initial()

    result = manager.reload()

    assert result.success is True
    assert result.previous_release_id == "release-1"
    assert result.active_release_id == "release-2"
    assert result.error is None
    assert manager.get_bundle() is second_bundle
    assert serving_readiness() == 1


def test_failed_reload_preserves_previous_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_mock_bundles(monkeypatch)
    initial_bundle = bundle("release-1")
    loader = MagicMock(
        side_effect=[
            initial_bundle,
            RuntimeError("invalid challenger"),
        ]
    )
    manager = ModelManager(loader)
    manager.load_initial()

    result = manager.reload()

    assert result.success is False
    assert result.previous_release_id == "release-1"
    assert result.active_release_id == "release-1"
    assert result.error == "invalid challenger"
    assert manager.get_bundle() is initial_bundle
    assert manager.last_reload_error == "invalid challenger"
    assert serving_readiness() == 1


def test_failed_first_reload_keeps_manager_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_mock_bundles(monkeypatch)
    manager = ModelManager(
        MagicMock(
            side_effect=RuntimeError("no active release")
        )
    )

    result = manager.reload()

    assert result.success is False
    assert result.previous_release_id is None
    assert result.active_release_id is None
    assert manager.ready is False
    assert serving_readiness() == 0


def test_replace_bundle_sets_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_mock_bundles(monkeypatch)
    replacement = bundle("release-manual")
    manager = ModelManager(MagicMock())

    result = manager.replace_bundle(replacement)

    assert result is replacement
    assert manager.get_bundle() is replacement
    assert manager.active_release_id == "release-manual"
    assert serving_readiness() == 1


def test_successful_reload_clears_previous_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_mock_bundles(monkeypatch)
    recovered_bundle = bundle("release-recovered")
    loader = MagicMock(
        side_effect=[
            RuntimeError("temporary failure"),
            recovered_bundle,
        ]
    )
    manager = ModelManager(loader)

    failed_result = manager.reload()
    successful_result = manager.reload()

    assert failed_result.success is False
    assert successful_result.success is True
    assert manager.last_reload_error is None
    assert manager.get_bundle() is recovered_bundle
    assert serving_readiness() == 1