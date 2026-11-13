import glob
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ..configs.paths import is_object_storage_path


def _gcs_fs() -> Any:
    """Return a GCS filesystem instance."""
    try:
        import gcsfs
    except ImportError as exc:
        raise RuntimeError("gcsfs is required for gs:// path operations.") from exc

    return gcsfs.GCSFileSystem()


def _s3_fs() -> Any:
    """Return an S3 filesystem using the default credential chain."""
    try:
        import s3fs
    except ImportError as exc:
        raise RuntimeError("S3 support is missing. Run: uv sync --extra aws") from exc

    return s3fs.S3FileSystem()


def _remote_fs(path: str) -> Any:
    """Select the filesystem for a supported object-storage URI."""
    if path.startswith("gs://"):
        return _gcs_fs()

    if path.startswith("s3://"):
        return _s3_fs()

    raise ValueError(f"Unsupported object-storage path: {path}")


def _parse_modified_time(value: Any) -> float:
    """Convert a filesystem timestamp value to Unix seconds."""
    if value is None:
        return 0.0

    if hasattr(value, "timestamp"):
        return float(value.timestamp())

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        normalized_value = value.replace("Z", "+00:00")

        try:
            return datetime.fromisoformat(normalized_value).timestamp()
        except ValueError:
            return 0.0

    return 0.0


def file_exists(path: str) -> bool:
    """Check whether a local or object-storage path exists."""
    normalized_path = str(path)

    if is_object_storage_path(normalized_path):
        return bool(_remote_fs(normalized_path).exists(normalized_path))

    return Path(normalized_path).exists()


def ensure_dir(path: str) -> None:
    """Create a local directory; object-storage prefixes need no creation."""
    normalized_path = str(path)

    if is_object_storage_path(normalized_path):
        return

    Path(normalized_path).mkdir(parents=True, exist_ok=True)


def list_files(path_pattern: str) -> list[str]:
    """List files matching a local or object-storage glob pattern."""
    normalized_pattern = str(path_pattern)

    if is_object_storage_path(normalized_pattern):
        files = _remote_fs(normalized_pattern).glob(normalized_pattern)
        scheme = normalized_pattern.split("://", maxsplit=1)[0]
        prefix = f"{scheme}://"

        return sorted(
            str(path) if str(path).startswith(prefix) else f"{prefix}{path}" for path in files
        )

    return sorted(glob.glob(normalized_pattern))


def modified_time(path: str) -> float:
    """Return a comparable modification time for a local or remote file."""
    normalized_path = str(path)

    if is_object_storage_path(normalized_path):
        info = _remote_fs(normalized_path).info(normalized_path)
        value = (
            info.get("LastModified")
            or info.get("updated")
            or info.get("mtime")
            or info.get("created")
            or info.get("timeCreated")
        )
        return _parse_modified_time(value)

    return Path(normalized_path).stat().st_mtime


def remove_file(path: str) -> None:
    """Remove a local or object-storage file if it exists."""
    normalized_path = str(path)

    if is_object_storage_path(normalized_path):
        filesystem = _remote_fs(normalized_path)

        if filesystem.exists(normalized_path):
            filesystem.rm(normalized_path)

        return

    local_path = Path(normalized_path)

    if local_path.exists():
        local_path.unlink()


def read_text(path: str) -> str:
    """Read UTF-8 text from a local or object-storage path."""
    normalized_path = str(path)

    if is_object_storage_path(normalized_path):
        with _remote_fs(normalized_path).open(normalized_path, "r", encoding="utf-8") as file:
            return str(file.read())

    return Path(normalized_path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    """Write UTF-8 text, replacing local files atomically."""
    normalized_path = str(path)

    if is_object_storage_path(normalized_path):
        with _remote_fs(normalized_path).open(normalized_path, "w", encoding="utf-8") as file:
            file.write(text)

        return

    local_path = Path(normalized_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=local_path.parent,
            prefix=f".{local_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, local_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
