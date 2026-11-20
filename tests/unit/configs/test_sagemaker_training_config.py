from pathlib import Path

import pytest

from mlops_sales_forecasting.aws.training_config import build_training_job_config


def prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        "model:\n"
        "  type: xgboost\n"
        "  params:\n"
        "    n_estimators: 20\n"
        "tracking:\n"
        "  mlflow_tracking_uri: http://localhost:5000\n"
        "services:\n"
        "  prefect_api_url: http://localhost:4200/api\n"
        "notifications:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "train.csv").write_text("example\n", encoding="utf-8")
    (input_directory / "store.csv").write_text("example\n", encoding="utf-8")
    return config_path, input_directory


def test_job_config_is_isolated_from_cloud_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, input_directory = prepare_inputs(tmp_path)
    work_directory = tmp_path / "work"
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("S3_BUCKET_NAME", "unrelated-bucket")

    config = build_training_job_config(
        config_path,
        input_directory=input_directory,
        work_directory=work_directory,
    )

    assert config["model"]["params"]["n_estimators"] == 20
    assert config["paths"]["raw_data"] == str(input_directory.resolve())
    assert config["paths"]["models"] == str(work_directory.resolve() / "models")
    assert config["notifications"] == {"enabled": False}
    assert "tracking" not in config
    assert "services" not in config
    assert not work_directory.exists()


@pytest.mark.parametrize("missing_file", ["train.csv", "store.csv"])
def test_job_config_requires_input_files(
    tmp_path: Path,
    missing_file: str,
) -> None:
    config_path, input_directory = prepare_inputs(tmp_path)
    (input_directory / missing_file).unlink()

    with pytest.raises(FileNotFoundError, match="Training input is missing"):
        build_training_job_config(
            config_path,
            input_directory=input_directory,
            work_directory=tmp_path / "work",
        )


@pytest.mark.parametrize("relationship", ["same", "work_inside_input", "input_inside_work"])
def test_job_config_rejects_overlapping_directories(
    tmp_path: Path,
    relationship: str,
) -> None:
    config_path, input_directory = prepare_inputs(tmp_path)

    if relationship == "same":
        work_directory = input_directory
    elif relationship == "work_inside_input":
        work_directory = input_directory / "work"
    else:
        work_directory = tmp_path

    with pytest.raises(ValueError, match="must not overlap"):
        build_training_job_config(
            config_path,
            input_directory=input_directory,
            work_directory=work_directory,
        )
