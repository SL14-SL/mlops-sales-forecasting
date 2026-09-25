from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.orchestration import (
    training_flow as training_flow_module,
)


def test_training_flow_builds_and_executes_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "project": {
            "task_type": "forecasting",
        },
    }
    pipeline = MagicMock()
    expected_result = MagicMock()

    load_config = MagicMock(
        return_value=config
    )
    build_pipeline = MagicMock(
        return_value=pipeline
    )
    execute_lifecycle = MagicMock(
        return_value=expected_result
    )

    monkeypatch.setattr(
        training_flow_module,
        "load_config",
        load_config,
    )
    monkeypatch.setattr(
        training_flow_module,
        "build_project_training_pipeline",
        build_pipeline,
    )
    monkeypatch.setattr(
        training_flow_module,
        "run_prefect_model_lifecycle",
        execute_lifecycle,
    )

    result = training_flow_module.training_flow.fn(
        run_id="pipeline-run-123"
    )

    assert result is expected_result
    load_config.assert_called_once_with()
    build_pipeline.assert_called_once_with(
        config
    )
    execute_lifecycle.assert_called_once_with(
        pipeline=pipeline,
        pipeline_run_id="pipeline-run-123",
    )


def test_training_flow_allows_generated_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock()

    monkeypatch.setattr(
        training_flow_module,
        "load_config",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        training_flow_module,
        "build_project_training_pipeline",
        MagicMock(return_value=pipeline),
    )

    execute_lifecycle = MagicMock()
    monkeypatch.setattr(
        training_flow_module,
        "run_prefect_model_lifecycle",
        execute_lifecycle,
    )

    training_flow_module.training_flow.fn()

    execute_lifecycle.assert_called_once_with(
        pipeline=pipeline,
        pipeline_run_id=None,
    )


def test_training_flow_propagates_extension_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_error = NotImplementedError(
        "Project pipeline is not implemented."
    )

    monkeypatch.setattr(
        training_flow_module,
        "load_config",
        MagicMock(
            return_value={
                "project": {
                    "task_type": "forecasting",
                },
            }
        ),
    )
    monkeypatch.setattr(
        training_flow_module,
        "build_project_training_pipeline",
        MagicMock(
            side_effect=extension_error
        ),
    )

    execute_lifecycle = MagicMock()
    monkeypatch.setattr(
        training_flow_module,
        "run_prefect_model_lifecycle",
        execute_lifecycle,
    )

    with pytest.raises(
        NotImplementedError,
        match="Project pipeline is not implemented",
    ):
        training_flow_module.training_flow.fn()

    execute_lifecycle.assert_not_called()


def test_flow_uses_project_specific_name() -> None:
    assert (
        training_flow_module.training_flow.name
        == "mlops-sales-forecasting-training"
    )