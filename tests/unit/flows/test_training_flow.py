import os
from unittest.mock import MagicMock, call, patch

import pytest

os.environ["PREFECT_API_MODE"] = "ephemeral"
os.environ.pop("PREFECT_API_URL", None)

import flows.training_flow as training_flow


@pytest.fixture(autouse=True)
def mock_flow_runtime():
    test_env_cfg = {
        "environment": "test",
        "api": {
            "url": "http://testserver/predict",
        },
        "services": {
            "prefect_api_url": "http://testserver/api",
        },
    }

    mock_logger = MagicMock()

    with (
        patch(
            "flows.training_flow.ENV_CFG",
            test_env_cfg,
        ),
        patch(
            "flows.training_flow.get_run_logger",
            return_value=mock_logger,
        ),
    ):
        yield


def test_training_pipeline_stable_system_only_evaluates_champion(
    monkeypatch,
):
    mock_check_drift = MagicMock(return_value=False)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock()
    mock_train = MagicMock()
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock()
    mock_final_refit = MagicMock()
    mock_refresh_api = MagicMock()
    mock_verify_release = MagicMock()

    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_final_refit",
        mock_final_refit,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_refresh_api",
        mock_refresh_api,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_verify_serving_release",
        mock_verify_release,
    )

    result = training_flow.training_pipeline.fn(
        force_run=False,
    )

    assert result is None

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_called_once()

    mock_prepare_data.assert_not_called()
    mock_snapshot_dataset.assert_not_called()
    mock_train.assert_not_called()
    mock_log_dataset_metadata.assert_not_called()
    mock_eval_and_reg.assert_not_called()
    mock_final_refit.assert_not_called()
    mock_refresh_api.assert_not_called()
    mock_verify_release.assert_not_called()


def test_training_pipeline_force_run_without_promotion(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds_test_001",
    }

    mock_check_drift = MagicMock(return_value=False)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest,
    )
    mock_train = MagicMock(return_value="run_123")
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(return_value=False)
    mock_final_refit = MagicMock()
    mock_refresh_api = MagicMock()
    mock_verify_release = MagicMock(return_value=True)
    mock_publish_release = MagicMock()
    mock_resolve_previous_release = MagicMock()
    mock_deploy_and_verify = MagicMock()

    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_final_refit",
        mock_final_refit,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_refresh_api",
        mock_refresh_api,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_verify_serving_release",
        mock_verify_release,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        training_flow,
        "task_resolve_previous_release",
        mock_resolve_previous_release,
    )
    monkeypatch.setattr(
        training_flow,
        "deploy_and_verify_release",
        mock_deploy_and_verify,
    )
    result = training_flow.training_pipeline.fn(
        force_run=True,
    )
    

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(
        is_drift_run=False,
    )
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once_with(
        is_drift_run=False,
    )
    mock_log_dataset_metadata.assert_called_once_with(
        "run_123",
        dataset_manifest,
    )
    mock_eval_and_reg.assert_called_once_with(
        "run_123",
    )
    mock_final_refit.assert_not_called()

    mock_refresh_api.assert_not_called()
    mock_verify_release.assert_not_called()
    mock_publish_release.assert_not_called()
    mock_resolve_previous_release.assert_not_called()
    mock_deploy_and_verify.assert_not_called()

    assert result == {
        "run_id": "run_123",
        "candidate_run_id": "run_123",
        "final_refit_run_id": None,
        "model_version": None,
        "release_id": None,
        "champion_promoted": False,
        "deployment": None,
    }


def test_training_pipeline_drift_with_final_refit(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds_test_002",
    }
    deployment_result = {
        "deployment_status": "verified",
        "release_id": "release-test-v8",
        "verification": {
            "release_id": "release-test-v8",
        },
        "rolled_back": False,
    }

    mock_resolve_previous_release = MagicMock(
        return_value="release-test-v7",
    )
    mock_deploy_and_verify = MagicMock(
        return_value=deployment_result,
    )

    mock_check_drift = MagicMock(return_value=True)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest,
    )
    mock_train = MagicMock(return_value="candidate_run_456")
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(return_value=True)
    mock_final_refit = MagicMock(
        return_value={
            "run_id": "final_run_456",
            "model_version": "8",
        },
    )

    mock_publish_release = MagicMock(
        return_value="release-test-v8",
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_final_refit",
        mock_final_refit,
    )
    monkeypatch.setattr(
        training_flow,
        "task_resolve_previous_release",
        mock_resolve_previous_release,
    )
    monkeypatch.setattr(
        training_flow,
        "deploy_and_verify_release",
        mock_deploy_and_verify,
    )

    result = training_flow.training_pipeline.fn(
        force_run=False,
    )

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(
        is_drift_run=True,
    )
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once_with(
        is_drift_run=True,
    )
    mock_eval_and_reg.assert_called_once_with(
        "candidate_run_456",
    )
    mock_final_refit.assert_called_once_with(
        candidate_run_id="candidate_run_456",
        is_drift_run=True,
    )

    mock_publish_release.assert_called_once_with(
        final_run_id="final_run_456",
        model_version="8",
        dataset_manifest=dataset_manifest,
    )

    assert mock_log_dataset_metadata.call_args_list == [
        call(
            "candidate_run_456",
            dataset_manifest,
        ),
        call(
            "final_run_456",
            dataset_manifest,
        ),
    ]

    mock_resolve_previous_release.assert_called_once_with()

    mock_deploy_and_verify.assert_called_once_with(
        release_id="release-test-v8",
        previous_release_id="release-test-v7",
    )

    assert result == {
        "run_id": "final_run_456",
        "candidate_run_id": (
            "candidate_run_456"
        ),
        "final_refit_run_id": (
            "final_run_456"
        ),
        "model_version": "8",
        "release_id": "release-test-v8",
        "champion_promoted": True,
        "deployment": deployment_result,
    }


def test_training_pipeline_drift_without_final_refit(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds_test_003",
    }

    mock_check_drift = MagicMock(return_value=True)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest,
    )
    mock_train = MagicMock(return_value="candidate_run_789")
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(return_value=False)
    mock_final_refit = MagicMock()
    mock_refresh_api = MagicMock()
    mock_verify_release = MagicMock(return_value=True)
    mock_publish_release = MagicMock()
    mock_resolve_previous_release = MagicMock()
    mock_deploy_and_verify = MagicMock()


    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_final_refit",
        mock_final_refit,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_refresh_api",
        mock_refresh_api,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_verify_serving_release",
        mock_verify_release,
    )
    monkeypatch.setattr(
        training_flow,
        "task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        training_flow,
        "task_resolve_previous_release",
        mock_resolve_previous_release,
    )
    monkeypatch.setattr(
        training_flow,
        "deploy_and_verify_release",
        mock_deploy_and_verify,
    )

    result = training_flow.training_pipeline.fn(
        force_run=False,
    )

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(
        is_drift_run=True,
    )
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once_with(
        is_drift_run=True,
    )
    mock_log_dataset_metadata.assert_called_once_with(
        "candidate_run_789",
        dataset_manifest,
    )
    mock_eval_and_reg.assert_called_once_with(
        "candidate_run_789",
    )
    mock_final_refit.assert_not_called()

    mock_refresh_api.assert_not_called()
    mock_verify_release.assert_not_called()
    mock_publish_release.assert_not_called()
    mock_resolve_previous_release.assert_not_called()
    mock_deploy_and_verify.assert_not_called()

    assert result == {
        "run_id": "candidate_run_789",
        "candidate_run_id": (
            "candidate_run_789"
        ),
        "final_refit_run_id": None,
        "model_version": None,
        "release_id": None,
        "champion_promoted": False,
        "deployment": None,
    }

def test_candidate_evaluation_error_does_not_register_or_promote(
    monkeypatch,
):
    comparison_error = RuntimeError(
        "Champion evaluation unavailable"
    )

    mock_compare_models = MagicMock(
        side_effect=comparison_error,
    )
    mock_register_model = MagicMock()

    monkeypatch.setattr(
        training_flow,
        "compare_models",
        mock_compare_models,
    )
    monkeypatch.setattr(
        training_flow,
        "register_model",
        mock_register_model,
    )

    with pytest.raises(
        RuntimeError,
        match="Champion evaluation unavailable",
    ):
        training_flow.task_eval_and_reg.fn(
            "candidate-run-123",
        )

    mock_compare_models.assert_called_once_with(
        "candidate-run-123",
    )
    mock_register_model.assert_not_called()

def test_bootstrap_rejected_when_champion_exists(
    monkeypatch,
):
    monkeypatch.setattr(
        training_flow,
        "champion_exists",
        MagicMock(return_value=True),
    )

    with pytest.raises(
        RuntimeError,
        match="Champion already exists",
    ):
        training_flow.training_pipeline.fn(
            force_run=True,
            bootstrap=True,
        )

def test_bootstrap_creates_initial_final_refit_champion(
    monkeypatch,
):
    champion_checks = MagicMock(
        side_effect=[False, False],
    )

    mock_train = MagicMock(
        return_value=(
            MagicMock(),
            "final-run-123",
        ),
    )

    registered_version = MagicMock()
    registered_version.version = "1"

    mock_register = MagicMock(
        return_value=registered_version,
    )

    monkeypatch.setattr(
        training_flow,
        "champion_exists",
        champion_checks,
    )
    monkeypatch.setattr(
        training_flow,
        "train",
        mock_train,
    )
    monkeypatch.setattr(
        training_flow,
        "register_model",
        mock_register,
    )

    result = (
        training_flow
        .task_bootstrap_champion
        .fn(
            candidate_run_id="candidate-run-123",
            is_drift_run=False,
        )
    )

    assert result == {
        "run_id": "final-run-123",
        "model_version": "1",
    }

    assert champion_checks.call_count == 2

    mock_train.assert_called_once_with(
        is_drift_run=False,
        run_role="final_refit",
        candidate_run_id="candidate-run-123",
    )

    mock_register.assert_called_once_with(
        "final-run-123",
        alias="champion",
    )


def test_publication_failure_does_not_reload_api(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds-test",
        "git_commit": "abc123",
        "snapshots": {
            "validated_store": (
                "data/versioning/"
                "ds-test/validated/store.parquet"
            ),
        },
    }
    
    monkeypatch.setattr(
        training_flow,
        "task_check_drift",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        training_flow,
        "task_prepare_data",
        MagicMock(),
    )
    monkeypatch.setattr(
        training_flow,
        "task_snapshot_dataset",
        MagicMock(
            return_value=dataset_manifest
        ),
    )
    monkeypatch.setattr(
        training_flow,
        "task_train",
        MagicMock(
            return_value="candidate-run"
        ),
    )
    monkeypatch.setattr(
        training_flow,
        "task_log_dataset_metadata",
        MagicMock(),
    )
    monkeypatch.setattr(
        training_flow,
        "task_eval_and_reg",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        training_flow,
        "task_final_refit",
        MagicMock(
            return_value={
                "run_id": "final-run",
                "model_version": "9",
            }
        ),
    )

    publication_error = RuntimeError(
        "GCS release publication failed"
    )

    monkeypatch.setattr(
        training_flow,
        "task_publish_serving_release",
        MagicMock(
            side_effect=publication_error
        ),
    )

    mock_resolve_previous_release = MagicMock(
        return_value="release-previous-v8",
    )
    mock_deploy_and_verify = MagicMock()

    monkeypatch.setattr(
        training_flow,
        "task_resolve_previous_release",
        mock_resolve_previous_release,
    )
    monkeypatch.setattr(
        training_flow,
        "deploy_and_verify_release",
        mock_deploy_and_verify,
    )

    with pytest.raises(
        RuntimeError,
        match="GCS release publication failed",
    ):
        training_flow.training_pipeline.fn(
            force_run=False,
        )

    mock_resolve_previous_release.assert_called_once_with()
    mock_deploy_and_verify.assert_not_called()


def test_training_pipeline_bootstrap_publishes_release(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds-bootstrap-001",
        "git_commit": "abc123",
    }
    deployment_result = {
        "deployment_status": "verified",
        "release_id": "release-bootstrap-v1",
        "verification": {
            "release_id": "release-bootstrap-v1",
        },
        "rolled_back": False,
    }

    mock_resolve_previous_release = MagicMock(
        return_value=None,
    )
    mock_deploy_and_verify = MagicMock(
        return_value=deployment_result,
    )

    mock_champion_exists = MagicMock(
        return_value=False,
    )
    mock_check_drift = MagicMock(
        return_value=False,
    )
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest,
    )
    mock_train = MagicMock(
        return_value="candidate-bootstrap-run",
    )
    mock_log_dataset_metadata = MagicMock()
    mock_bootstrap = MagicMock(
        return_value={
            "run_id": "final-bootstrap-run",
            "model_version": "1",
        },
    )
    mock_publish = MagicMock(
        return_value="release-bootstrap-v1",
    )

    monkeypatch.setattr(
        training_flow,
        "champion_exists",
        mock_champion_exists,
    )
    monkeypatch.setattr(
        training_flow,
        "task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        training_flow,
        "task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        training_flow,
        "task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        training_flow,
        "task_train",
        mock_train,
    )
    monkeypatch.setattr(
        training_flow,
        "task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        training_flow,
        "task_bootstrap_champion",
        mock_bootstrap,
    )
    monkeypatch.setattr(
        training_flow,
        "task_publish_serving_release",
        mock_publish,
    )
    monkeypatch.setattr(
        training_flow,
        "task_resolve_previous_release",
        mock_resolve_previous_release,
    )
    monkeypatch.setattr(
        training_flow,
        "deploy_and_verify_release",
        mock_deploy_and_verify,
    )

    result = training_flow.training_pipeline.fn(
        force_run=True,
        bootstrap=True,
    )

    mock_bootstrap.assert_called_once_with(
        candidate_run_id=(
            "candidate-bootstrap-run"
        ),
        is_drift_run=False,
    )

    mock_publish.assert_called_once_with(
        final_run_id="final-bootstrap-run",
        model_version="1",
        dataset_manifest=dataset_manifest,
    )

    mock_resolve_previous_release.assert_called_once_with()

    mock_deploy_and_verify.assert_called_once_with(
        release_id="release-bootstrap-v1",
        previous_release_id=None,
    )

    assert result == {
        "run_id": "final-bootstrap-run",
        "candidate_run_id": (
            "candidate-bootstrap-run"
        ),
        "final_refit_run_id": (
            "final-bootstrap-run"
        ),
        "model_version": "1",
        "release_id": (
            "release-bootstrap-v1"
        ),
        "champion_promoted": True,
        "deployment": deployment_result,
    }
