from unittest.mock import MagicMock

from mlops_sales_forecasting.orchestration import (
    auto_retraining_flow as flow_module,
)
from mlops_sales_forecasting.orchestration.retraining_service import (
    AutoRetrainingResult,
)


def test_flow_loads_config_and_runs_service(
    monkeypatch,
) -> None:
    config = {
        "paths": {
            "monitoring": "data/monitoring",
        },
    }
    expected = AutoRetrainingResult(
        status="retrained",
        decision_id="retrain-test-123",
        reasons=("Persistent degradation.",),
        candidate_run_id="mlflow-run-123",
        champion_promoted=True,
    )

    load_config = MagicMock(return_value=config)
    run_service = MagicMock(return_value=expected)
    logger = MagicMock()

    monkeypatch.setattr(
        flow_module,
        "load_config",
        load_config,
    )
    monkeypatch.setattr(
        flow_module,
        "run_auto_retraining",
        run_service,
    )
    monkeypatch.setattr(
        flow_module,
        "get_run_logger",
        MagicMock(return_value=logger),
    )

    result = flow_module.auto_retraining_flow.fn()

    assert result == {
        "status": "retrained",
        "decision_id": ("retrain-test-123"),
        "reasons": [
            "Persistent degradation.",
        ],
        "candidate_run_id": ("mlflow-run-123"),
        "champion_promoted": True,
    }

    load_config.assert_called_once_with()
    run_service.assert_called_once_with(config=config)
    logger.info.assert_called_once()


def test_flow_uses_project_specific_name() -> None:
    assert flow_module.auto_retraining_flow.name == ("mlops-sales-forecasting-auto-retraining")
