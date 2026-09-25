import glob
from datetime import datetime
from pathlib import Path
from typing import Any


def _gcs_fs() -> Any:
    """Return a GCS filesystem instance when GCS support is installed."""
    try:
        import gcsfs
    except ImportError as exc:
        raise RuntimeError(
            "gcsfs is required for gs:// path operations."
        ) from exc

    return gcsfs.GCSFileSystem()


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
    """Check whether a local or GCS path exists."""
    normalized_path = str(path)

    if normalized_path.startswith("gs://"):
        return bool(_gcs_fs().exists(normalized_path))

    return Path(normalized_path).exists()


def ensure_dir(path: str) -> None:
    """Create a local directory if it does not already exist."""
    normalized_path = str(path)

    if normalized_path.startswith("gs://"):
        return

    Path(normalized_path).mkdir(parents=True, exist_ok=True)


def list_files(path_pattern: str) -> list[str]:
    """List files matching a local or GCS glob pattern."""
    normalized_pattern = str(path_pattern)

    if normalized_pattern.startswith("gs://"):
        files = _gcs_fs().glob(normalized_pattern)

        return sorted(
            str(path) if str(path).startswith("gs://") else f"gs://{path}"
            for path in files
        )

    return sorted(glob.glob(normalized_pattern))


def modified_time(path: str) -> float:
    """Return a comparable modification time for a local or GCS path."""
    normalized_path = str(path)

    if normalized_path.startswith("gs://"):
        info = _gcs_fs().info(normalized_path)
        value = (
            info.get("updated")
            or info.get("mtime")
            or info.get("created")
            or info.get("timeCreated")
        )
        return _parse_modified_time(value)

    return Path(normalized_path).stat().st_mtime


def remove_file(path: str) -> None:
    """Remove a local or GCS file if it exists."""
    normalized_path = str(path)

    if normalized_path.startswith("gs://"):
        filesystem = _gcs_fs()

        if filesystem.exists(normalized_path):
            filesystem.rm(normalized_path)

        return

    local_path = Path(normalized_path)

    if local_path.exists():
        local_path.unlink()


def read_text(path: str) -> str:
    """Read UTF-8 text from a local or GCS path."""
    normalized_path = str(path)

    if normalized_path.startswith("gs://"):
        with _gcs_fs().open(normalized_path, "r") as file:
            return str(file.read())

    return Path(normalized_path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    """Write UTF-8 text to a local or GCS path."""
    normalized_path = str(path)

    if normalized_path.startswith("gs://"):
        with _gcs_fs().open(normalized_path, "w") as file:
            file.write(text)

        return

    local_path = Path(normalized_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text, encoding="utf-8")