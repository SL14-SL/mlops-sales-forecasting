from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.inference import serving
from mlops_sales_forecasting.inference.releases.contracts import (
    TaskType,
)


def build_config() -> dict:
    return {
        "paths": {
            "models": "artifacts/models",
        },
        "tracking": {
            "mlflow_tracking_uri": "http://localhost:5000",
        },
        "serving": {
            "alias": "champion",
        },
    }


def test_expected_task_type() -> None:
    assert serving.EXPECTED_TASK_TYPE is TaskType.FORECASTING


def test_serving_settings_from_config() -> None:
    result = serving.serving_settings_from_config(
        build_config()
    )

    assert result.models_path == "artifacts/models"
    assert result.tracking_uri == "http://localhost:5000"
    assert result.serving_alias == "champion"
    assert result.task_type is TaskType.FORECASTING


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("paths", "models"),
        ("tracking", "mlflow_tracking_uri"),
        ("serving", "alias"),
    ],
)
def test_serving_settings_require_values(
    section: str,
    field: str,
) -> None:
    config = build_config()
    del config[section][field]

    with pytest.raises(
        ValueError,
        match="Missing or invalid serving configuration",
    ):
        serving.serving_settings_from_config(config)


def test_load_active_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_mlflow = MagicMock()
    expected_bundle = MagicMock()
    expected_bundle.manifest.task_type = (
        TaskType.FORECASTING
    )
    load_bundle = MagicMock(
        return_value=expected_bundle
    )

    monkeypatch.setattr(
        serving,
        "configure_mlflow",
        configure_mlflow,
    )
    monkeypatch.setattr(
        serving,
        "load_active_serving_bundle",
        load_bundle,
    )

    result = serving.load_active_bundle(build_config())

    assert result is expected_bundle
    configure_mlflow.assert_called_once_with(
        "http://localhost:5000"
    )
    assert load_bundle.call_args.kwargs["models_path"] == (
        "artifacts/models"
    )
    assert load_bundle.call_args.kwargs["serving_alias"] == (
        "champion"
    )


def test_load_bundle_for_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        serving,
        "configure_mlflow",
        MagicMock(),
    )

    expected_bundle = MagicMock()
    expected_bundle.manifest.task_type = (
        TaskType.FORECASTING
    )
    load_bundle = MagicMock(
        return_value=expected_bundle
    )
    monkeypatch.setattr(
        serving,
        "load_serving_bundle",
        load_bundle,
    )

    result = serving.load_bundle_for_release(
        build_config(),
        release_id="release-5",
    )

    assert result is expected_bundle
    assert load_bundle.call_args.kwargs["release_id"] == (
        "release-5"
    )


def test_load_active_bundle_rejects_wrong_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        serving,
        "configure_mlflow",
        MagicMock(),
    )

    bundle = MagicMock()

    bundle.manifest.task_type = TaskType.CLASSIFICATION

    monkeypatch.setattr(
        serving,
        "load_active_serving_bundle",
        MagicMock(return_value=bundle),
    )

    with pytest.raises(
        ValueError,
        match="unexpected task type",
    ):
        serving.load_active_bundle(build_config())