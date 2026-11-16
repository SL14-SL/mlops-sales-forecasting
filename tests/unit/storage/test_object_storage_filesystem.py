from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.storage import filesystem


@pytest.fixture(params=["gs", "s3"])
def remote_storage(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, MagicMock]:
    scheme = request.param
    remote = MagicMock()
    factory_name = "_gcs_fs" if scheme == "gs" else "_s3_fs"
    monkeypatch.setattr(filesystem, factory_name, lambda: remote)
    return scheme, remote


def test_remote_file_exists(
    remote_storage: tuple[str, MagicMock],
) -> None:
    scheme, remote = remote_storage
    path = f"{scheme}://bucket/data.csv"
    remote.exists.return_value = False

    assert filesystem.file_exists(path) is False
    remote.exists.assert_called_once_with(path)


def test_remote_directory_is_not_created(
    remote_storage: tuple[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheme, remote = remote_storage
    mkdir = MagicMock()
    monkeypatch.setattr(filesystem.Path, "mkdir", mkdir)

    filesystem.ensure_dir(f"{scheme}://bucket/prefix")

    mkdir.assert_not_called()
    remote.mkdir.assert_not_called()


def test_remote_listing_preserves_scheme_and_sorts(
    remote_storage: tuple[str, MagicMock],
) -> None:
    scheme, remote = remote_storage
    pattern = f"{scheme}://bucket/data/*.csv"
    remote.glob.return_value = [
        "bucket/data/b.csv",
        f"{scheme}://bucket/data/a.csv",
    ]

    assert filesystem.list_files(pattern) == [
        f"{scheme}://bucket/data/a.csv",
        f"{scheme}://bucket/data/b.csv",
    ]
    remote.glob.assert_called_once_with(pattern)


def test_remote_modified_time(
    remote_storage: tuple[str, MagicMock],
) -> None:
    scheme, remote = remote_storage
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    key = "LastModified" if scheme == "s3" else "updated"
    remote.info.return_value = {key: timestamp}

    assert filesystem.modified_time(f"{scheme}://bucket/data.csv") == timestamp.timestamp()


def test_remote_text_round_trip_calls(
    remote_storage: tuple[str, MagicMock],
) -> None:
    scheme, remote = remote_storage
    path = f"{scheme}://bucket/example.txt"
    handle = remote.open.return_value.__enter__.return_value
    handle.read.return_value = "Grüße"

    filesystem.write_text(path, "Grüße")
    assert filesystem.read_text(path) == "Grüße"

    handle.write.assert_called_once_with("Grüße")
    remote.open.assert_any_call(path, "w", encoding="utf-8")
    remote.open.assert_any_call(path, "r", encoding="utf-8")


@pytest.mark.parametrize("exists", [True, False])
def test_remote_removal(
    remote_storage: tuple[str, MagicMock],
    exists: bool,
) -> None:
    scheme, remote = remote_storage
    path = f"{scheme}://bucket/example.txt"
    remote.exists.return_value = exists

    filesystem.remove_file(path)

    if exists:
        remote.rm.assert_called_once_with(path)
    else:
        remote.rm.assert_not_called()
