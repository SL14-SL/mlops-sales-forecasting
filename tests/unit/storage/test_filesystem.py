from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.storage import filesystem


def test_file_exists_for_local_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("content", encoding="utf-8")

    assert filesystem.file_exists(str(file_path)) is True
    assert filesystem.file_exists(str(tmp_path / "missing.txt")) is False


def test_ensure_dir_creates_nested_local_directory(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "directory"

    filesystem.ensure_dir(str(directory))

    assert directory.is_dir()


def test_list_files_returns_sorted_local_matches(tmp_path: Path) -> None:
    (tmp_path / "b.csv").write_text("b", encoding="utf-8")
    (tmp_path / "a.csv").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("text", encoding="utf-8")

    result = filesystem.list_files(str(tmp_path / "*.csv"))

    assert result == [
        str(tmp_path / "a.csv"),
        str(tmp_path / "b.csv"),
    ]


def test_modified_time_returns_local_timestamp(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("content", encoding="utf-8")

    assert filesystem.modified_time(str(file_path)) == pytest.approx(
        file_path.stat().st_mtime
    )


def test_write_text_creates_parent_directories(tmp_path: Path) -> None:
    file_path = tmp_path / "nested" / "example.txt"

    filesystem.write_text(str(file_path), "example content")

    assert file_path.read_text(encoding="utf-8") == "example content"


def test_read_text_reads_local_utf8_content(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("Grüße aus dem Test", encoding="utf-8")

    assert filesystem.read_text(str(file_path)) == "Grüße aus dem Test"


def test_remove_file_removes_existing_local_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("content", encoding="utf-8")

    filesystem.remove_file(str(file_path))

    assert file_path.exists() is False


def test_remove_file_ignores_missing_local_file(tmp_path: Path) -> None:
    filesystem.remove_file(str(tmp_path / "missing.txt"))


def test_parse_modified_time_accepts_datetime() -> None:
    value = datetime(2026, 1, 1, tzinfo=UTC)

    assert filesystem._parse_modified_time(value) == value.timestamp()


def test_parse_modified_time_accepts_iso_timestamp() -> None:
    result = filesystem._parse_modified_time("2026-01-01T12:00:00Z")

    expected = datetime(2026, 1, 1, 12, 0, tzinfo=UTC).timestamp()
    assert result == expected


def test_parse_modified_time_returns_zero_for_invalid_value() -> None:
    assert filesystem._parse_modified_time("not-a-timestamp") == 0.0
    assert filesystem._parse_modified_time(None) == 0.0

def test_file_exists_uses_gcs_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_filesystem = MagicMock()
    fake_filesystem.exists.return_value = True
    monkeypatch.setattr(filesystem, "_gcs_fs", lambda: fake_filesystem)

    result = filesystem.file_exists("gs://example-bucket/data.csv")

    assert result is True
    fake_filesystem.exists.assert_called_once_with(
        "gs://example-bucket/data.csv"
    )


def test_ensure_dir_does_nothing_for_gcs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gcs_factory = MagicMock()
    monkeypatch.setattr(filesystem, "_gcs_fs", gcs_factory)

    filesystem.ensure_dir("gs://example-bucket/prefix")

    gcs_factory.assert_not_called()


def test_list_files_normalizes_gcs_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_filesystem = MagicMock()
    fake_filesystem.glob.return_value = [
        "example-bucket/data/b.csv",
        "gs://example-bucket/data/a.csv",
    ]
    monkeypatch.setattr(filesystem, "_gcs_fs", lambda: fake_filesystem)

    result = filesystem.list_files("gs://example-bucket/data/*.csv")

    assert result == [
        "gs://example-bucket/data/a.csv",
        "gs://example-bucket/data/b.csv",
    ]


def test_modified_time_reads_gcs_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_filesystem = MagicMock()
    fake_filesystem.info.return_value = {
        "updated": "2026-01-01T12:00:00Z"
    }
    monkeypatch.setattr(filesystem, "_gcs_fs", lambda: fake_filesystem)

    result = filesystem.modified_time("gs://example-bucket/data.csv")

    expected = datetime(2026, 1, 1, 12, 0, tzinfo=UTC).timestamp()
    assert result == expected


def test_read_text_uses_gcs_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_file = MagicMock()
    fake_file.read.return_value = "remote content"

    context_manager = MagicMock()
    context_manager.__enter__.return_value = fake_file

    fake_filesystem = MagicMock()
    fake_filesystem.open.return_value = context_manager
    monkeypatch.setattr(filesystem, "_gcs_fs", lambda: fake_filesystem)

    result = filesystem.read_text("gs://example-bucket/example.txt")

    assert result == "remote content"
    fake_filesystem.open.assert_called_once_with(
        "gs://example-bucket/example.txt",
        "r",
    )


def test_write_text_uses_gcs_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_file = MagicMock()

    context_manager = MagicMock()
    context_manager.__enter__.return_value = fake_file

    fake_filesystem = MagicMock()
    fake_filesystem.open.return_value = context_manager
    monkeypatch.setattr(filesystem, "_gcs_fs", lambda: fake_filesystem)

    filesystem.write_text(
        "gs://example-bucket/example.txt",
        "remote content",
    )

    fake_filesystem.open.assert_called_once_with(
        "gs://example-bucket/example.txt",
        "w",
    )
    fake_file.write.assert_called_once_with("remote content")


def test_remove_file_removes_existing_gcs_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_filesystem = MagicMock()
    fake_filesystem.exists.return_value = True
    monkeypatch.setattr(filesystem, "_gcs_fs", lambda: fake_filesystem)

    filesystem.remove_file("gs://example-bucket/example.txt")

    fake_filesystem.rm.assert_called_once_with(
        "gs://example-bucket/example.txt"
    )


def test_remove_file_ignores_missing_gcs_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_filesystem = MagicMock()
    fake_filesystem.exists.return_value = False
    monkeypatch.setattr(filesystem, "_gcs_fs", lambda: fake_filesystem)

    filesystem.remove_file("gs://example-bucket/missing.txt")

    fake_filesystem.rm.assert_not_called()