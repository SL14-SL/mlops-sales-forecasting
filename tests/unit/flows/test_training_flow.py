import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["PREFECT_API_MODE"] = "ephemeral"
os.environ.pop("PREFECT_API_URL", None)

import flows.training_flow as training_flow


@pytest.fixture(autouse=True)
def mock_flow_runtime():
    test_env_cfg = {
        "environment": "test",
        "api": {"url": "http://testserver/predict"},
        "services": {"prefect_api_url": "http://testserver/api"},
    }

    mock_logger = MagicMock()

    with (
        patch("flows.training_flow.ENV_CFG", test_env_cfg),
        patch("flows.training_flow.get_run_logger", return_value=mock_logger),
        patch("flows.training_flow.task_refresh_api", return_value=None),
        patch("flows.training_flow.task_verify_health", return_value=True),
    ):
        yield


def test_training_pipeline_stable_system_only_evaluates_champion(monkeypatch):
    mock_check_drift = MagicMock(return_value=False)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock()
    mock_train = MagicMock()
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock()
    mock_refresh_api = MagicMock()
    mock_verify_health = MagicMock()

    monkeypatch.setattr("flows.training_flow.task_check_drift", mock_check_drift)
    monkeypatch.setattr("flows.training_flow.task_evaluate_champion", mock_evaluate_champion)
    monkeypatch.setattr("flows.training_flow.task_prepare_data", mock_prepare_data)
    monkeypatch.setattr("flows.training_flow.task_snapshot_dataset", mock_snapshot_dataset)
    monkeypatch.setattr("flows.training_flow.task_train", mock_train)
    monkeypatch.setattr("flows.training_flow.task_log_dataset_metadata", mock_log_dataset_metadata)
    monkeypatch.setattr("flows.training_flow.task_eval_and_reg", mock_eval_and_reg)
    monkeypatch.setattr("flows.training_flow.task_refresh_api", mock_refresh_api)
    monkeypatch.setattr("flows.training_flow.task_verify_health", mock_verify_health)

    training_flow.training_pipeline.fn(force_run=False)

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_called_once()

    mock_prepare_data.assert_not_called()
    mock_snapshot_dataset.assert_not_called()
    mock_train.assert_not_called()
    mock_log_dataset_metadata.assert_not_called()
    mock_eval_and_reg.assert_not_called()
    mock_refresh_api.assert_not_called()
    mock_verify_health.assert_not_called()


def test_training_pipeline_force_run_executes_training_path(monkeypatch):
    mock_check_drift = MagicMock(return_value=False)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(return_value={"dataset_version": "ds_test_001"})
    mock_train = MagicMock(return_value="run_123")
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(return_value=False)
    mock_refresh_api = MagicMock()
    mock_verify_health = MagicMock()

    monkeypatch.setattr("flows.training_flow.task_check_drift", mock_check_drift)
    monkeypatch.setattr("flows.training_flow.task_evaluate_champion", mock_evaluate_champion)
    monkeypatch.setattr("flows.training_flow.task_prepare_data", mock_prepare_data)
    monkeypatch.setattr("flows.training_flow.task_snapshot_dataset", mock_snapshot_dataset)
    monkeypatch.setattr("flows.training_flow.task_train", mock_train)
    monkeypatch.setattr("flows.training_flow.task_log_dataset_metadata", mock_log_dataset_metadata)
    monkeypatch.setattr("flows.training_flow.task_eval_and_reg", mock_eval_and_reg)
    monkeypatch.setattr("flows.training_flow.task_refresh_api", mock_refresh_api)
    monkeypatch.setattr("flows.training_flow.task_verify_health", mock_verify_health)

    training_flow.training_pipeline.fn(force_run=True)

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(is_drift_run=False)
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once()
    mock_log_dataset_metadata.assert_called_once_with("run_123", {"dataset_version": "ds_test_001"})
    mock_eval_and_reg.assert_called_once_with("run_123")

    mock_refresh_api.assert_called_once()
    mock_verify_health.assert_called_once()


def test_training_pipeline_drift_with_new_champion_refreshes_api(monkeypatch):
    mock_check_drift = MagicMock(return_value=True)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(return_value={"dataset_version": "ds_test_002"})
    mock_train = MagicMock(return_value="run_456")
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(return_value=True)
    mock_refresh_api = MagicMock()
    mock_verify_health = MagicMock()

    monkeypatch.setattr("flows.training_flow.task_check_drift", mock_check_drift)
    monkeypatch.setattr("flows.training_flow.task_evaluate_champion", mock_evaluate_champion)
    monkeypatch.setattr("flows.training_flow.task_prepare_data", mock_prepare_data)
    monkeypatch.setattr("flows.training_flow.task_snapshot_dataset", mock_snapshot_dataset)
    monkeypatch.setattr("flows.training_flow.task_train", mock_train)
    monkeypatch.setattr("flows.training_flow.task_log_dataset_metadata", mock_log_dataset_metadata)
    monkeypatch.setattr("flows.training_flow.task_eval_and_reg", mock_eval_and_reg)
    monkeypatch.setattr("flows.training_flow.task_refresh_api", mock_refresh_api)
    monkeypatch.setattr("flows.training_flow.task_verify_health", mock_verify_health)

    training_flow.training_pipeline.fn(force_run=False)

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(is_drift_run=True)
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once()
    mock_log_dataset_metadata.assert_called_once_with("run_456", {"dataset_version": "ds_test_002"})
    mock_eval_and_reg.assert_called_once_with("run_456")

    mock_refresh_api.assert_called_once()
    mock_verify_health.assert_called_once()


def test_training_pipeline_drift_without_new_champion_refreshes_api(monkeypatch):
    mock_check_drift = MagicMock(return_value=True)
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(return_value={"dataset_version": "ds_test_003"})
    mock_train = MagicMock(return_value="run_789")
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(return_value=False)
    mock_refresh_api = MagicMock()
    mock_verify_health = MagicMock()

    monkeypatch.setattr("flows.training_flow.task_check_drift", mock_check_drift)
    monkeypatch.setattr("flows.training_flow.task_evaluate_champion", mock_evaluate_champion)
    monkeypatch.setattr("flows.training_flow.task_prepare_data", mock_prepare_data)
    monkeypatch.setattr("flows.training_flow.task_snapshot_dataset", mock_snapshot_dataset)
    monkeypatch.setattr("flows.training_flow.task_train", mock_train)
    monkeypatch.setattr("flows.training_flow.task_log_dataset_metadata", mock_log_dataset_metadata)
    monkeypatch.setattr("flows.training_flow.task_eval_and_reg", mock_eval_and_reg)
    monkeypatch.setattr("flows.training_flow.task_refresh_api", mock_refresh_api)
    monkeypatch.setattr("flows.training_flow.task_verify_health", mock_verify_health)

    training_flow.training_pipeline.fn(force_run=False)

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(is_drift_run=True)
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once()
    mock_log_dataset_metadata.assert_called_once_with("run_789", {"dataset_version": "ds_test_003"})
    mock_eval_and_reg.assert_called_once_with("run_789")

    mock_refresh_api.assert_called_once()
    mock_verify_health.assert_called_once()