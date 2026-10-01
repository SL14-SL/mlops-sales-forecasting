from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.orchestration import (
    lifecycle_adapter,
)
from mlops_sales_forecasting.pipeline.service import (
    TrainingPipeline,
)


def test_promoted_candidate_publishes_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(spec=TrainingPipeline)
    pipeline.config = {
        "project": {
            "slug": "lifecycle-release-test",
        },
        "environment": "test",
        "notifications": {
            "enabled": False,
        },
        "paths": {
            "models": "artifacts/models",
        },
        "tracking": {
            "mlflow_tracking_uri": ("http://localhost:5000"),
            "experiment_name": "release-test",
            "model_name": "release-test-model",
        },
    }
    pipeline.model_logger = MagicMock()
    pipeline.release_input_provider = MagicMock()
    pipeline.evaluator = MagicMock()

    start_training_run = MagicMock()
    start_training_run.return_value.__enter__.return_value = "mlflow-run-7"

    tracked_result = MagicMock()
    training_result = MagicMock()
    evaluation_result = MagicMock()
    tracked_result.pipeline.training = training_result
    tracked_result.pipeline.evaluation = evaluation_result

    run_pipeline = MagicMock(return_value=tracked_result)
    log_training = MagicMock()
    log_evaluation = MagicMock()

    model_artifact = MagicMock()
    log_artifact = MagicMock(return_value=model_artifact)

    promotion = MagicMock()
    promotion.decision.promote = True
    promotion.decision.metric_name = "rmse"
    promotion.decision.candidate_value = 0.81
    promotion.decision.champion_value = 0.85
    promotion.decision.improvement = 0.04
    promotion.decision.reason = "Candidate satisfies policy."
    promotion.previous_champion_version = "6"

    registration = MagicMock()
    registration.registered = True
    registration.run_id = "mlflow-run-7"
    registration.model_name = "release-test-model"
    registration.model_version = "7"
    registration.model_uri = "models:/release-test-model/7"

    candidate_result = MagicMock()
    candidate_result.promotion = promotion
    candidate_result.registration = registration

    finalize_candidate = MagicMock(return_value=candidate_result)

    release_input = MagicMock()
    release_input.task_type = MagicMock()
    release_input.model_type = "xgboost"
    release_input.sources = {
        "feature_schema": MagicMock(),
    }
    release_input.metadata = {
        "decision_threshold": 0.42,
    }
    release_input.dataset_version = "dataset-v3"
    release_input.config_hash = "config-hash"
    release_input.git_commit = "abc123"

    build_release_input = MagicMock(return_value=release_input)

    published_release = MagicMock()
    published_release.manifest.release_id = "release-7"
    published_release.release_root = "artifacts/models/releases/release-7"
    publish_release = MagicMock(return_value=published_release)
    restore_champion = MagicMock()

    monkeypatch.setattr(
        lifecycle_adapter,
        "restore_champion",
        restore_champion,
    )

    monkeypatch.setattr(
        lifecycle_adapter,
        "start_training_run",
        start_training_run,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "run_prefect_training_pipeline",
        run_pipeline,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "log_training_result",
        log_training,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "log_evaluation_result",
        log_evaluation,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "log_model_artifact",
        log_artifact,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "finalize_configured_model_candidate",
        finalize_candidate,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "build_serving_release_input",
        build_release_input,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "publish_serving_release",
        publish_release,
    )

    result = lifecycle_adapter.run_prefect_model_lifecycle.fn(
        pipeline=pipeline,
    )

    assert result.serving_release is published_release

    build_release_input.assert_called_once_with(
        provider=(pipeline.release_input_provider),
        training_result=training_result,
        evaluation_result=evaluation_result,
        config=pipeline.config,
    )
    publish_release.assert_called_once_with(
        models_path="artifacts/models",
        registration=(candidate_result.registration),
        promotion=promotion,
        task_type=release_input.task_type,
        model_type="xgboost",
        sources=release_input.sources,
        metadata=release_input.metadata,
        dataset_version="dataset-v3",
        config_hash="config-hash",
        git_commit="abc123",
    )
    restore_champion.assert_not_called()


def test_unpromoted_candidate_skips_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_result = MagicMock()
    candidate_result.promotion = MagicMock()
    candidate_result.promotion.decision.promote = False

    build_release_input = MagicMock()
    publish_release = MagicMock()

    monkeypatch.setattr(
        lifecycle_adapter,
        "build_serving_release_input",
        build_release_input,
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "publish_serving_release",
        publish_release,
    )

    result = lifecycle_adapter._publish_promoted_release(
        pipeline=MagicMock(),
        tracked_result=MagicMock(),
        candidate_result=candidate_result,
    )

    assert result is None
    build_release_input.assert_not_called()
    publish_release.assert_not_called()


def test_promoted_release_requires_models_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock()
    pipeline.config = {
        "paths": {},
    }
    pipeline.release_input_provider = MagicMock()

    tracked_result = MagicMock()
    candidate_result = MagicMock()
    candidate_result.registration = MagicMock()
    candidate_result.promotion.decision.promote = True
    candidate_result.promotion.previous_champion_version = "6"

    release_input = MagicMock()
    release_input.task_type = MagicMock()
    release_input.model_type = "xgboost"
    release_input.sources = {
        "schema": MagicMock(),
    }
    release_input.metadata = {}
    release_input.dataset_version = None
    release_input.config_hash = None
    release_input.git_commit = None

    monkeypatch.setattr(
        lifecycle_adapter,
        "build_serving_release_input",
        MagicMock(return_value=release_input),
    )

    restore_champion = MagicMock()
    monkeypatch.setattr(
        lifecycle_adapter,
        "restore_champion",
        restore_champion,
    )

    with pytest.raises(
        ValueError,
        match="Config path 'models'",
    ):
        lifecycle_adapter._publish_promoted_release(
            pipeline=pipeline,
            tracked_result=tracked_result,
            candidate_result=candidate_result,
        )

    restore_champion.assert_called_once_with(
        registration=(candidate_result.registration),
        previous_champion_version="6",
        config=pipeline.config,
    )


def test_release_failure_restores_previous_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(spec=TrainingPipeline)
    pipeline.config = {
        "paths": {
            "models": "artifacts/models",
        },
        "tracking": {
            "mlflow_tracking_uri": ("http://localhost:5000"),
        },
    }
    pipeline.release_input_provider = MagicMock()

    tracked_result = MagicMock()
    candidate_result = MagicMock()
    candidate_result.registration = MagicMock()

    promotion = MagicMock()
    promotion.decision.promote = True
    promotion.previous_champion_version = "6"
    candidate_result.promotion = promotion

    release_input = MagicMock()
    release_input.task_type = MagicMock()
    release_input.model_type = "xgboost"
    release_input.sources = {
        "schema": MagicMock(),
    }
    release_input.metadata = {}
    release_input.dataset_version = None
    release_input.config_hash = None
    release_input.git_commit = None

    monkeypatch.setattr(
        lifecycle_adapter,
        "build_serving_release_input",
        MagicMock(return_value=release_input),
    )

    release_error = RuntimeError("release publication failed")
    monkeypatch.setattr(
        lifecycle_adapter,
        "publish_serving_release",
        MagicMock(side_effect=release_error),
    )

    restore_champion = MagicMock()
    monkeypatch.setattr(
        lifecycle_adapter,
        "restore_champion",
        restore_champion,
    )

    with pytest.raises(
        RuntimeError,
        match="release publication failed",
    ):
        lifecycle_adapter._publish_promoted_release(
            pipeline=pipeline,
            tracked_result=tracked_result,
            candidate_result=candidate_result,
        )

    restore_champion.assert_called_once_with(
        registration=(candidate_result.registration),
        previous_champion_version="6",
        config=pipeline.config,
    )


def test_failed_initial_release_removes_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(spec=TrainingPipeline)
    pipeline.config = {
        "paths": {
            "models": "artifacts/models",
        },
    }
    pipeline.release_input_provider = MagicMock()

    tracked_result = MagicMock()
    candidate_result = MagicMock()
    candidate_result.registration = MagicMock()

    promotion = MagicMock()
    promotion.decision.promote = True
    promotion.previous_champion_version = None
    candidate_result.promotion = promotion

    release_input = MagicMock()
    release_input.task_type = MagicMock()
    release_input.model_type = "xgboost"
    release_input.sources = {
        "schema": MagicMock(),
    }
    release_input.metadata = {}
    release_input.dataset_version = None
    release_input.config_hash = None
    release_input.git_commit = None

    monkeypatch.setattr(
        lifecycle_adapter,
        "build_serving_release_input",
        MagicMock(return_value=release_input),
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "publish_serving_release",
        MagicMock(side_effect=RuntimeError("release failed")),
    )

    restore_champion = MagicMock()
    monkeypatch.setattr(
        lifecycle_adapter,
        "restore_champion",
        restore_champion,
    )

    with pytest.raises(
        RuntimeError,
        match="release failed",
    ):
        lifecycle_adapter._publish_promoted_release(
            pipeline=pipeline,
            tracked_result=tracked_result,
            candidate_result=candidate_result,
        )

    restore_champion.assert_called_once_with(
        registration=(candidate_result.registration),
        previous_champion_version=None,
        config=pipeline.config,
    )


def test_reports_release_and_restoration_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MagicMock(spec=TrainingPipeline)
    pipeline.config = {
        "paths": {
            "models": "artifacts/models",
        },
    }
    pipeline.release_input_provider = MagicMock()

    tracked_result = MagicMock()
    candidate_result = MagicMock()
    candidate_result.registration = MagicMock()

    promotion = MagicMock()
    promotion.decision.promote = True
    promotion.previous_champion_version = "6"
    candidate_result.promotion = promotion

    release_input = MagicMock()
    release_input.task_type = MagicMock()
    release_input.model_type = "xgboost"
    release_input.sources = {
        "schema": MagicMock(),
    }
    release_input.metadata = {}
    release_input.dataset_version = None
    release_input.config_hash = None
    release_input.git_commit = None

    release_error = RuntimeError("release failed")
    restoration_error = RuntimeError("restoration failed")

    monkeypatch.setattr(
        lifecycle_adapter,
        "build_serving_release_input",
        MagicMock(return_value=release_input),
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "publish_serving_release",
        MagicMock(side_effect=release_error),
    )
    monkeypatch.setattr(
        lifecycle_adapter,
        "restore_champion",
        MagicMock(side_effect=restoration_error),
    )

    with pytest.raises(
        ExceptionGroup,
        match="both failed",
    ) as captured:
        lifecycle_adapter._publish_promoted_release(
            pipeline=pipeline,
            tracked_result=tracked_result,
            candidate_result=candidate_result,
        )

    assert captured.value.exceptions == (
        release_error,
        restoration_error,
    )
