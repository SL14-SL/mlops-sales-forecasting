from pathlib import Path

from mlops_sales_forecasting.configs.paths import (
    get_project_root,
    join_uri,
    path_name,
    path_suffix,
)


def test_get_project_root_contains_pyproject() -> None:
    project_root = get_project_root()

    assert isinstance(project_root, Path)
    assert (project_root / "pyproject.toml").is_file()


def test_join_uri_joins_local_path() -> None:
    result = join_uri("/tmp/data/", "/processed", "train.parquet")

    assert result == "/tmp/data/processed/train.parquet"


def test_join_uri_preserves_gcs_prefix() -> None:
    result = join_uri("gs://example-bucket/", "/data", "train.parquet")

    assert result == "gs://example-bucket/data/train.parquet"


def test_join_uri_ignores_empty_parts() -> None:
    result = join_uri("gs://example-bucket/", "", "data", "")

    assert result == "gs://example-bucket/data"


def test_path_name_supports_local_and_remote_paths() -> None:
    assert path_name("/tmp/data/train.parquet") == "train.parquet"
    assert path_name("gs://example-bucket/data/train.parquet") == "train.parquet"


def test_path_suffix_is_lowercase() -> None:
    assert path_suffix("/tmp/data/TRAIN.PARQUET") == ".parquet"
    assert path_suffix("gs://example-bucket/data/train.CSV") == ".csv"