from unittest.mock import MagicMock

import pytest

from flows.tasks import registry_tasks


@pytest.fixture(autouse=True)
def mock_prefect_runtime(
    monkeypatch,
):
    mock_logger = MagicMock()

    monkeypatch.setattr(
        registry_tasks,
        "get_run_logger",
        MagicMock(
            return_value=mock_logger,
        ),
    )

    return mock_logger

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
        registry_tasks,
        "compare_models",
        mock_compare_models,
    )
    monkeypatch.setattr(
        registry_tasks,
        "register_model",
        mock_register_model,
    )

    with pytest.raises(
        RuntimeError,
        match="Champion evaluation unavailable",
    ):
        registry_tasks.task_eval_and_reg.fn(
            "candidate-run-123",
        )

    mock_compare_models.assert_called_once_with(
        "candidate-run-123",
    )
    mock_register_model.assert_not_called()

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
        registry_tasks,
        "champion_exists",
        champion_checks,
    )
    monkeypatch.setattr(
        registry_tasks,
        "train",
        mock_train,
    )
    monkeypatch.setattr(
        registry_tasks,
        "register_model",
        mock_register,
    )

    result = (
        registry_tasks
        .task_bootstrap_champion
        .fn(
            candidate_run_id=(
                "candidate-run-123"
            ),
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
        candidate_run_id=(
            "candidate-run-123"
        ),
    )

    mock_register.assert_called_once_with(
        "final-run-123",
        alias="champion",
    )
