from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.tracking import mlflow as tracking
from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)


def build_config() -> dict:
    return {
        "tracking": {
            "mlflow_tracking_uri": (
                "http://localhost:5000"
            ),
            "experiment_name": (
                "example-training-dev"
            ),
        },
    }


def build_training_result(
    run_id: str = "mlflow-run-123",
) -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id=run_id,
        metrics={
            "validation_score": 0.82,
        },
        parameters={
            "max_depth": 5,
        },
        artifacts={
            "feature_schema": (
                "artifacts/feature-schema.json"
            ),
        },
    )


def test_load_tracking_settings() -> None:
    settings = (
        tracking
        .load_mlflow_tracking_settings(
            build_config()
        )
    )

    assert settings.tracking_uri == (
        "http://localhost:5000"
    )
    assert settings.experiment_name == (
        "example-training-dev"
    )


@pytest.mark.parametrize(
    "tracking_section",
    [
        None,
        [],
        "invalid",
    ],
)
def test_tracking_section_must_be_mapping(
    tracking_section: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="'tracking' section",
    ):
        tracking.load_mlflow_tracking_settings(
            {
                "tracking": tracking_section,
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "mlflow_tracking_uri",
        "experiment_name",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        7,
        None,
        "${UNRESOLVED}",
    ],
)
def test_tracking_values_must_be_resolved(
    field: str,
    value: object,
) -> None:
    config = build_config()
    config["tracking"][field] = value

    with pytest.raises(
        ValueError,
        match=field,
    ):
        tracking.load_mlflow_tracking_settings(
            config
        )


def test_configure_mlflow_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_tracking_uri = MagicMock()
    set_experiment = MagicMock()

    monkeypatch.setattr(
        tracking.mlflow,
        "set_tracking_uri",
        set_tracking_uri,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "set_experiment",
        set_experiment,
    )

    settings = (
        tracking
        .load_mlflow_tracking_settings(
            build_config()
        )
    )
    tracking.configure_mlflow_tracking(
        settings
    )

    set_tracking_uri.assert_called_once_with(
        "http://localhost:5000"
    )
    set_experiment.assert_called_once_with(
        "example-training-dev"
    )


def test_start_training_run_yields_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "mlflow-run-123"

    run_context = MagicMock()
    run_context.__enter__.return_value = (
        active_run
    )

    start_run = MagicMock(
        return_value=run_context
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "start_run",
        start_run,
    )
    monkeypatch.setattr(
        tracking,
        "configure_mlflow_tracking",
        MagicMock(),
    )

    with tracking.start_training_run(
        build_config(),
        run_name="candidate-training",
        tags={
            "environment": "test",
        },
    ) as run_id:
        assert run_id == "mlflow-run-123"

    start_run.assert_called_once_with(
        run_name="candidate-training",
        tags={
            "environment": "test",
        },
    )
    run_context.__exit__.assert_called_once()


def test_logging_requires_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking.mlflow,
        "active_run",
        MagicMock(return_value=None),
    )

    with pytest.raises(
        RuntimeError,
        match="active MLflow run",
    ):
        tracking.log_training_result(
            build_training_result()
        )


def test_result_run_id_must_match_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "different-run"

    monkeypatch.setattr(
        tracking.mlflow,
        "active_run",
        MagicMock(return_value=active_run),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        tracking.log_training_result(
            build_training_result()
        )


def test_log_training_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "mlflow-run-123"

    monkeypatch.setattr(
        tracking.mlflow,
        "active_run",
        MagicMock(return_value=active_run),
    )

    log_params = MagicMock()
    log_metrics = MagicMock()
    set_tags = MagicMock()

    monkeypatch.setattr(
        tracking.mlflow,
        "log_params",
        log_params,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "log_metrics",
        log_metrics,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "set_tags",
        set_tags,
    )

    result = build_training_result()

    returned = tracking.log_training_result(
        result
    )

    assert returned is result
    log_params.assert_called_once_with(
        {
            "max_depth": 5,
        }
    )
    log_metrics.assert_called_once_with(
        {
            "validation_score": 0.82,
        }
    )
    set_tags.assert_called_once_with(
        {
            "artifact_uri.feature_schema": (
                "artifacts/feature-schema.json"
            ),
        }
    )


def test_empty_collections_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_run = MagicMock()
    active_run.info.run_id = "mlflow-run-123"

    monkeypatch.setattr(
        tracking.mlflow,
        "active_run",
        MagicMock(return_value=active_run),
    )

    log_params = MagicMock()
    log_metrics = MagicMock()
    set_tags = MagicMock()

    monkeypatch.setattr(
        tracking.mlflow,
        "log_params",
        log_params,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "log_metrics",
        log_metrics,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "set_tags",
        set_tags,
    )

    tracking.log_training_result(
        TrainingResult(
            model=object(),
            run_id="mlflow-run-123",
            metrics={},
        )
    )

    log_params.assert_not_called()
    log_metrics.assert_not_called()
    set_tags.assert_not_called()


def test_logging_rejects_wrong_result_type() -> None:
    with pytest.raises(
        TypeError,
        match="requires TrainingResult",
    ):
        tracking.log_training_result(
            object()  # type: ignore[arg-type]
        )