from pathlib import Path, PurePosixPath


def get_project_root() -> Path:
    """Return the project directory containing pyproject.toml."""
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise RuntimeError("Could not determine project root.")


def join_uri(base: str, *parts: str) -> str:
    """Join local or remote URI components without breaking the URI prefix."""
    normalized_base = str(base).rstrip("/")
    normalized_parts = [str(part).strip("/") for part in parts]
    suffix = "/".join(part for part in normalized_parts if part)

    if not suffix:
        return normalized_base

    return f"{normalized_base}/{suffix}"


def path_name(path: str) -> str:
    """Return the final component of a local path or object-storage URI."""
    return PurePosixPath(str(path)).name


def path_suffix(path: str) -> str:
    """Return the lowercase suffix of a local path or object-storage URI."""
    return PurePosixPath(str(path)).suffix.lower()