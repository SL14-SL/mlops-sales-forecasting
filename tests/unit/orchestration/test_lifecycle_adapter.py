from unittest.mock import MagicMock, patch

import pytest

from mlops_sales_forecasting.notifications.contracts import (
    LifecycleEventType,
)
from mlops_sales_forecasting.orchestration.lifecycle_adapter import (
    PrefectTrainingLifecycleResult,
    run_prefect_model_lifecycle,
)
from mlops_sales_forecasting.pipeline.service import (
    TrainingPipeline,
)


def build_pipeline() -> MagicMock:
    pipeline = MagicMock(
        spec=TrainingPipeline
    )
    pipeline.config = {
        "project": {
            "slug": "lifecycle-test",
        },
        "environment": "test",
        "notifications": {
            "enabled": False,
        },
        "tracking": {
            "mlflow_tracking_uri": (
                "http://localhost:5000"
            ),
            "experiment_name": "lifecycle-test",
            "model_name": "lifecycle-test-model",
        },
    }
    pipeline.model_logger = MagicMock()

    return pipeline


def rejected_candidate_result() -> MagicMock:
    registration = MagicMock()
    registration.registered = False
    registration.run_id = "run-rejected"
    registration.model_name = (
        "lifecycle-test-model"
    )
    registration.reason = (
        "Candidate did not pass the quality gate."
    )

    candidate = MagicMock()
    candidate.registration = registration
    candidate.promotion = None
    return candidate


@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_model_artifact"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "finalize_configured_model_candidate"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_evaluation_result"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_training_result"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "run_prefect_training_pipeline"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "start_training_run"
)
def test_runs_complete_model_lifecycle(
    start_training_run: MagicMock,
    run_prefect_training_pipeline: MagicMock,
    log_training_result: MagicMock,
    log_evaluation_result: MagicMock,
    finalize_candidate: MagicMock,
    log_model_artifact: MagicMock,
) -> None:
    pipeline = build_pipeline()

    start_training_run.return_value.__enter__.return_value = (
        "mlflow-run-1"
    )

    tracked_result = MagicMock()
    training_result = MagicMock()
    evaluation_result = MagicMock()

    tracked_result.pipeline.training = (
        training_result
    )
    tracked_result.pipeline.evaluation = (
        evaluation_result
    )
    run_prefect_training_pipeline.return_value = (
        tracked_result
    )

    candidate_result = (
        rejected_candidate_result()
    )
    model_artifact = MagicMock()

    log_model_artifact.return_value = (
        model_artifact
    )
    finalize_candidate.return_value = (
        candidate_result
    )

    result = run_prefect_model_lifecycle.fn(
        pipeline=pipeline,
        mlflow_run_name="scheduled-training",
        mlflow_tags={"trigger": "scheduled"},
    )

    assert isinstance(
        result,
        PrefectTrainingLifecycleResult,
    )
    assert result.pipeline is tracked_result
    assert (
        result.model_artifact
        is model_artifact
    )
    assert result.candidate is candidate_result
    assert result.serving_release is None

    start_training_run.assert_called_once_with(
        pipeline.config,
        run_name="scheduled-training",
        tags={"trigger": "scheduled"},
    )
    run_prefect_training_pipeline.assert_called_once_with(
        pipeline=pipeline,
        run_id="mlflow-run-1",
    )
    log_training_result.assert_called_once_with(
        training_result
    )
    log_evaluation_result.assert_called_once_with(
        evaluation_result
    )
    log_model_artifact.assert_called_once_with(
        logger=pipeline.model_logger,
        training_result=training_result,
        config=pipeline.config,
        artifact_path="model",
    )
    finalize_candidate.assert_called_once_with(
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=pipeline.config,
        artifact_path="model",
        logged_model_uri=(
            model_artifact.model_uri
        ),
    )


@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_model_artifact"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "finalize_configured_model_candidate"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_evaluation_result"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_training_result"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "run_prefect_training_pipeline"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "start_training_run"
)
def test_preserves_explicit_pipeline_run_id(
    start_training_run: MagicMock,
    run_prefect_training_pipeline: MagicMock,
    log_training_result: MagicMock,
    log_evaluation_result: MagicMock,
    finalize_candidate: MagicMock,
    log_model_artifact: MagicMock,
) -> None:
    pipeline = build_pipeline()

    start_training_run.return_value.__enter__.return_value = (
        "mlflow-run-2"
    )

    tracked_result = MagicMock()
    training_result = MagicMock()
    evaluation_result = MagicMock()

    tracked_result.pipeline.training = (
        training_result
    )
    tracked_result.pipeline.evaluation = (
        evaluation_result
    )
    run_prefect_training_pipeline.return_value = (
        tracked_result
    )

    model_artifact = MagicMock()
    candidate_result = (
        rejected_candidate_result()
    )

    log_model_artifact.return_value = (
        model_artifact
    )
    finalize_candidate.return_value = (
        candidate_result
    )
    notification_sink = MagicMock()

    result = run_prefect_model_lifecycle.fn(
        pipeline=pipeline,
        pipeline_run_id="pipeline-run-123",
        artifact_path="trained/model",
        notification_sink=notification_sink,
    )

    assert result.pipeline is tracked_result
    assert (
        result.model_artifact
        is model_artifact
    )
    assert result.candidate is candidate_result
    assert result.serving_release is None

    run_prefect_training_pipeline.assert_called_once_with(
        pipeline=pipeline,
        run_id="pipeline-run-123",
    )
    log_training_result.assert_called_once_with(
        training_result
    )
    log_evaluation_result.assert_called_once_with(
        evaluation_result
    )
    log_model_artifact.assert_called_once_with(
        logger=pipeline.model_logger,
        training_result=training_result,
        config=pipeline.config,
        artifact_path="trained/model",
    )
    finalize_candidate.assert_called_once_with(
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=pipeline.config,
        artifact_path="trained/model",
        logged_model_uri=(
            model_artifact.model_uri
        ),
    )
    notification_sink.notify.assert_called_once()

    event = (
        notification_sink
        .notify
        .call_args
        .args[0]
    )

    assert event.event_type == (
        LifecycleEventType.CANDIDATE_REJECTED
    )


@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_model_artifact"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "finalize_configured_model_candidate"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_evaluation_result"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "log_training_result"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "run_prefect_training_pipeline"
)
@patch(
    "mlops_sales_forecasting.orchestration."
    "lifecycle_adapter."
    "start_training_run"
)
def test_failed_pipeline_stops_model_lifecycle(
    start_training_run: MagicMock,
    run_prefect_training_pipeline: MagicMock,
    log_training_result: MagicMock,
    log_evaluation_result: MagicMock,
    finalize_candidate: MagicMock,
    log_model_artifact: MagicMock,
) -> None:
    pipeline = build_pipeline()

    start_training_run.return_value.__enter__.return_value = (
        "mlflow-run-3"
    )
    run_prefect_training_pipeline.side_effect = (
        RuntimeError("training failed")
    )
    notification_sink = MagicMock()

    with pytest.raises(
        RuntimeError,
        match="training failed",
    ):
        run_prefect_model_lifecycle.fn(
            pipeline=pipeline,
            notification_sink=notification_sink,
        )

    notification_sink.notify.assert_called_once()

    event = (
        notification_sink
        .notify
        .call_args
        .args[0]
    )

    assert event.event_type == (
        LifecycleEventType.PIPELINE_FAILED
    )
    assert event.run_id == "mlflow-run-3"
    assert event.details["error_message"] == (
        "training failed"
    )
    log_training_result.assert_not_called()
    log_evaluation_result.assert_not_called()
    log_model_artifact.assert_not_called()
    finalize_candidate.assert_not_called()