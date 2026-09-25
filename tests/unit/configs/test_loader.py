from pathlib import Path

import pytest

from mlops_sales_forecasting.configs import loader


def write_config(
    project_root: Path,
    filename: str,
    content: str,
) -> Path:
    config_directory = project_root / "configs"
    config_directory.mkdir(parents=True, exist_ok=True)

    config_path = config_directory / filename
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_yaml_returns_mapping(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        "example.yaml",
        "project:\n  name: example\n",
    )

    result = loader._load_yaml(config_path)

    assert result == {
        "project": {
            "name": "example",
        }
    }


def test_load_yaml_returns_empty_mapping_for_empty_file(
    tmp_path: Path,
) -> None:
    config_path = write_config(
        tmp_path,
        "empty.yaml",
        "",
    )

    assert loader._load_yaml(config_path) == {}


def test_load_yaml_rejects_non_mapping_content(
    tmp_path: Path,
) -> None:
    config_path = write_config(
        tmp_path,
        "invalid.yaml",
        "- first\n- second\n",
    )

    with pytest.raises(
        ValueError,
        match="must contain a YAML mapping",
    ):
        loader._load_yaml(config_path)


def test_load_yaml_rejects_missing_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Config file not found",
    ):
        loader._load_yaml(config_path)


def test_resolve_config_filename_uses_environment_default() -> None:
    result = loader._resolve_config_filename(None, "staging")

    assert result == "staging.yaml"


def test_resolve_config_filename_accepts_explicit_yaml_file() -> None:
    result = loader._resolve_config_filename(
        "training.yml",
        "dev",
    )

    assert result == "training.yml"


def test_resolve_config_filename_rejects_directory_components() -> None:
    with pytest.raises(
        ValueError,
        match="without directory components",
    ):
        loader._resolve_config_filename(
            "../prod.yaml",
            "dev",
        )


def test_resolve_config_filename_rejects_invalid_extension() -> None:
    with pytest.raises(
        ValueError,
        match="must use the .yaml or .yml extension",
    ):
        loader._resolve_config_filename(
            "dev.json",
            "dev",
        )


def test_load_config_uses_active_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(
        tmp_path,
        "staging.yaml",
        (
            "project:\n"
            "  name: example\n"
            "paths:\n"
            "  raw: data/raw\n"
        ),
    )

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.delenv("K_SERVICE", raising=False)

    result = loader.load_config()

    assert result["environment"] == "staging"
    assert result["project"]["name"] == "example"


def test_load_config_resolves_environment_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(
        tmp_path,
        "example.yaml",
        (
            "services:\n"
            "  endpoint: '${EXAMPLE_ENDPOINT}'\n"
        ),
    )

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(
        "EXAMPLE_ENDPOINT",
        "https://example.test",
    )

    result = loader.load_config("example.yaml")

    assert (
        result["services"]["endpoint"]
        == "https://example.test"
    )


def test_load_config_overrides_gcs_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(
        tmp_path,
        "example.yaml",
        (
            "paths:\n"
            "  raw: gs://old-bucket/data/raw\n"
        ),
    )

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("GCS_BUCKET_NAME", "new-bucket")

    result = loader.load_config("example.yaml")

    assert result["paths"]["raw"] == "gs://new-bucket/data/raw"


def test_get_path_returns_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(
        tmp_path,
        "example.yaml",
        (
            "paths:\n"
            "  processed: data/processed\n"
        ),
    )

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)

    result = loader.get_path(
        "processed",
        "example.yaml",
    )

    assert result == "data/processed"


def test_get_path_rejects_missing_paths_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(
        tmp_path,
        "example.yaml",
        "project:\n  name: example\n",
    )

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)

    with pytest.raises(
        KeyError,
        match="valid 'paths' section",
    ):
        loader.get_path("raw", "example.yaml")


def test_get_path_rejects_unknown_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(
        tmp_path,
        "example.yaml",
        "paths:\n  raw: data/raw\n",
    )

    monkeypatch.setattr(loader, "PROJECT_ROOT", tmp_path)

    with pytest.raises(
        KeyError,
        match="Path 'processed' not found",
    ):
        loader.get_path("processed", "example.yaml")