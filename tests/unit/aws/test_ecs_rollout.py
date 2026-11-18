from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.aws import ecs_rollout


def service_response(deployment_id="new", state="COMPLETED"):
    return {
        "services": [
            {
                "deployments": [
                    {
                        "id": deployment_id,
                        "status": "PRIMARY",
                        "rolloutState": state,
                    }
                ]
            }
        ]
    }


def test_waits_for_rollout_state_after_aws_waiter_would_report_stable(monkeypatch):
    ecs = MagicMock()
    ecs.describe_services.side_effect = [
        service_response(state="IN_PROGRESS"),
        service_response(state="IN_PROGRESS"),
        service_response(),
    ]
    sleeps = MagicMock()
    monkeypatch.setattr(ecs_rollout.time, "sleep", sleeps)
    ecs_rollout.wait_for_rollout(
        ecs=ecs, cluster="cluster", service="mlflow", deployment_id="new", max_attempts=3
    )
    assert ecs.describe_services.call_count == 3
    assert sleeps.call_count == 2


def test_waits_for_running_task_list_to_stop_returning_the_old_task(monkeypatch):
    ecs = MagicMock()
    ecs.describe_services.return_value = service_response()
    ecs.list_tasks.side_effect = [{"taskArns": ["old"]}, {"taskArns": ["replacement"]}]
    monkeypatch.setattr(ecs_rollout.time, "sleep", MagicMock())
    ecs_rollout.wait_for_rollout(
        ecs=ecs,
        cluster="cluster",
        service="mlflow",
        deployment_id="new",
        previous_task_arns=("old",),
        max_attempts=2,
    )
    assert ecs.list_tasks.call_count == 2


def test_failed_requested_deployment_is_not_hidden_by_polling(monkeypatch):
    ecs = MagicMock()
    ecs.describe_services.return_value = service_response(state="FAILED")
    sleeps = MagicMock()
    monkeypatch.setattr(ecs_rollout.time, "sleep", sleeps)
    with pytest.raises(RuntimeError, match="requested deployment failed"):
        ecs_rollout.wait_for_rollout(
            ecs=ecs, cluster="cluster", service="mlflow", deployment_id="new"
        )
    sleeps.assert_not_called()


def test_completed_old_deployment_is_not_accepted_after_circuit_rollback(monkeypatch):
    ecs = MagicMock()
    ecs.describe_services.return_value = service_response(deployment_id="old")
    monkeypatch.setattr(ecs_rollout.time, "sleep", MagicMock())
    with pytest.raises(RuntimeError, match="possible circuit rollback") as error:
        ecs_rollout.wait_for_rollout(
            ecs=ecs, cluster="cluster", service="mlflow", deployment_id="new", max_attempts=2
        )
    assert "'expected_deployment': 'new'" in str(error.value)
    assert "'primary_deployment': 'old'" in str(error.value)


def test_missing_service_is_reported_explicitly():
    ecs = MagicMock()
    ecs.describe_services.return_value = {"failures": [{"reason": "MISSING"}], "services": []}
    with pytest.raises(RuntimeError, match="service could not be read"):
        ecs_rollout.wait_for_rollout(
            ecs=ecs, cluster="cluster", service="mlflow", deployment_id="new"
        )
