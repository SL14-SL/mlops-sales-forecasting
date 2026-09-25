from pathlib import Path

import yaml


def load_deployment_config() -> dict:
    project_root = Path(
        __file__
    ).resolve().parents[3]

    with (
        project_root / "prefect.yaml"
    ).open(
        encoding="utf-8"
    ) as config_file:
        return yaml.safe_load(
            config_file
        )


def test_deployment_uses_project_name() -> None:
    config = load_deployment_config()

    assert config["name"] == (
        "mlops-sales-forecasting"
    )


def test_deployment_has_no_remote_steps() -> None:
    config = load_deployment_config()

    assert config["build"] == []
    assert config["push"] == []
    assert config["pull"] == []


def test_training_deployment_configuration() -> None:
    config = load_deployment_config()
    deployment = config["deployments"][0]

    assert deployment["name"] == (
        "local-training"
    )
    assert deployment["entrypoint"] == (
        "mlops_sales_forecasting."
        "orchestration.training_flow."
        "training_flow"
    )
    assert deployment["parameters"] == {}


def test_deployment_uses_local_process_pool() -> None:
    config = load_deployment_config()
    deployment = config["deployments"][0]
    work_pool = deployment["work_pool"]

    assert work_pool == {
        "name": "local-process-pool",
        "work_queue_name": "default",
        "job_variables": {},
    }


def test_deployment_has_task_type_tag() -> None:
    config = load_deployment_config()
    deployment = config["deployments"][0]

    assert "forecasting" in deployment["tags"]
    assert "training" in deployment["tags"]