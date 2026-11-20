import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from mlops_sales_forecasting.aws import simulation
from mlops_sales_forecasting.configs.paths import get_project_root


@pytest.mark.parametrize("days,mode", [(0, "both"), (96, "managed"), (5, "invalid")])
def test_invalid_simulation_options_fail_before_loading_credentials(monkeypatch, days, mode):
    def unexpected(**kwargs):
        pytest.fail("Invalid options must not load cloud configuration.")

    monkeypatch.setattr(simulation, "load_aws_lifecycle_config", unexpected)
    with pytest.raises(ValueError, match="Simulation"):
        simulation.run_aws_simulation(maximum_days=days, mode=mode)


def test_launcher_forwards_simulation_options_and_logs(monkeypatch, capsys):
    path = get_project_root() / "scripts/run_aws_task.py"
    spec = importlib.util.spec_from_file_location("simulation_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    outputs = {
        "fargate_training_cluster_arn": {"value": "cluster"},
        "fargate_training_task_definition_arn": {"value": "definition"},
        "training_subnet_id": {"value": "subnet"},
        "training_security_group_id": {"value": "security-group"},
        "fargate_training_log_group": {"value": "/ecs/training"},
    }
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(outputs)),
    )
    calls = []

    def aws(*args):
        calls.append(args)
        if args[:2] == ("ecs", "run-task"):
            return {"tasks": [{"taskArn": "arn:task/cluster/id"}]}
        if args[:2] == ("ecs", "describe-tasks"):
            return {"tasks": [{"lastStatus": "STOPPED", "containers": [{"exitCode": 0}]}]}
        assert args[:2] == ("logs", "get-log-events")
        return {"events": [{"message": "simulation result"}], "nextForwardToken": "token"}

    monkeypatch.setattr(launcher, "aws", aws)
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys, "argv", [str(path), "simulate", "--maximum-days", "90", "--simulation-mode", "both"]
    )
    launcher.main()
    launch = next(args for args in calls if args[:2] == ("ecs", "run-task"))
    override = json.loads(launch[launch.index("--overrides") + 1])
    assert override["containerOverrides"][0]["command"] == [
        "simulate",
        "--maximum-days",
        "90",
        "--simulation-mode",
        "both",
    ]
    assert "simulation result" in capsys.readouterr().out


def test_launcher_stops_simulation_when_wall_clock_deadline_passes(monkeypatch):
    path = get_project_root() / "scripts/run_aws_task.py"
    spec = importlib.util.spec_from_file_location("deadline_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    values = {
        key: {"value": key}
        for key in (
            "fargate_training_cluster_arn",
            "fargate_training_task_definition_arn",
            "training_subnet_id",
            "training_security_group_id",
            "fargate_training_log_group",
        )
    }
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(values)),
    )
    calls = []

    def aws(*args):
        calls.append(args)
        if args[:2] == ("ecs", "run-task"):
            return {"tasks": [{"taskArn": "arn:task/id"}]}
        assert args[:2] == ("ecs", "stop-task")
        return {}

    monkeypatch.setattr(launcher, "aws", aws)
    clock = iter([0, 50 * 60 + 1])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(sys, "argv", [str(path), "simulate"])
    with pytest.raises(TimeoutError):
        launcher.main()
    assert calls[-1][:2] == ("ecs", "stop-task")
