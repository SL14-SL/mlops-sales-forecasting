from pathlib import Path

import pytest

from mlops_sales_forecasting.configs.paths import is_object_storage_path


@pytest.mark.parametrize(
    "path",
    [
        "gs://example-bucket/data/train.csv",
        "s3://example-bucket/data/train.csv",
        "gs://example-bucket",
        "s3://example-bucket",
    ],
)
def test_recognizes_supported_object_storage_paths(path: str) -> None:
    assert is_object_storage_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/train.csv",
        "/tmp/data/train.csv",
        Path("data/raw/train.csv"),
        "https://example.com/train.csv",
        "s3-data/train.csv",
        "",
    ],
)
def test_other_paths_are_not_object_storage(path: str | Path) -> None:
    assert not is_object_storage_path(path)
