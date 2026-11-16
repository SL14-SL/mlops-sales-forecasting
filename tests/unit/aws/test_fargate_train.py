import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.aws import fargate_train


@pytest.fixture
def adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, MagicMock]:
    config_path = tmp_path / "training.yaml"
    config_path.write_text("model: {}", encoding="utf-8")
    client = MagicMock()

    def download(bucket: str, key: str, destination: str) -> None:
        Path(destination).write_text("synthetic input", encoding="utf-8")

    client.download_file.side_effect = download
    monkeypatch.setattr(fargate_train, "_s3_client", lambda: client)
    return config_path, client


def create_exports(command: list[str], **kwargs: object) -> None:
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 900

    input_directory = Path(command[command.index("--input-dir") + 1])
    assert (input_directory / "train.csv").read_text() == "synthetic input"
    assert (input_directory / "store.csv").read_text() == "synthetic input"

    model_directory = Path(command[command.index("--model-dir") + 1])
    output_directory = Path(command[command.index("--output-dir") + 1])

    for relative in (
        "model/MLmodel",
        "model/model.ubj",
        "latest_state.json",
        "known_calendar.parquet",
    ):
        path = model_directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("exported artifact", encoding="utf-8")

    output_directory.mkdir()
    for name in ("evaluation.json", "resolved_config.json"):
        (output_directory / name).write_text("{}", encoding="utf-8")


def test_uploads_exports_before_completion_marker(
    adapter: tuple[Path, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, client = adapter
    monkeypatch.setattr(fargate_train.subprocess, "run", create_exports)

    result_uri = fargate_train.run_fargate_training(
        input_uri="s3://input-bucket/training/input/v1/",
        output_uri="s3://output-bucket/training/output",
        config_path=config_path,
        estimators=3,
    )

    assert client.download_file.call_count == 2
    assert client.upload_file.call_count == 6
    assert result_uri.startswith("s3://output-bucket/training/output/fargate-")

    uploaded_keys = {call.args[2] for call in client.upload_file.call_args_list}
    assert any(key.endswith("/model/model/MLmodel") for key in uploaded_keys)
    assert any(key.endswith("/output/evaluation.json") for key in uploaded_keys)

    assert client.mock_calls[-1][0] == "put_object"
    completion = json.loads(client.put_object.call_args.kwargs["Body"])
    assert completion["status"] == "completed"
    assert completion["promoted"] is False
    assert completion["result_uri"] == result_uri


def test_failed_training_does_not_upload_results(
    adapter: tuple[Path, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, client = adapter

    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "training")

    monkeypatch.setattr(fargate_train.subprocess, "run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        fargate_train.run_fargate_training(
            input_uri="s3://input-bucket/data",
            output_uri="s3://output-bucket/results",
            config_path=config_path,
        )

    client.upload_file.assert_not_called()
    client.put_object.assert_not_called()


def test_failed_upload_does_not_publish_completion(
    adapter: tuple[Path, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, client = adapter
    monkeypatch.setattr(fargate_train.subprocess, "run", create_exports)
    client.upload_file.side_effect = RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        fargate_train.run_fargate_training(
            input_uri="s3://input-bucket/data",
            output_uri="s3://output-bucket/results",
            config_path=config_path,
        )

    client.put_object.assert_not_called()


@pytest.mark.parametrize(
    "uri",
    ["data/raw", "https://example.com/data", "s3:///data", "s3://bucket/data?x=1"],
)
def test_rejects_invalid_s3_prefixes(uri: str) -> None:
    with pytest.raises(ValueError, match="Expected an S3 prefix"):
        fargate_train._parse_s3_prefix(uri)
