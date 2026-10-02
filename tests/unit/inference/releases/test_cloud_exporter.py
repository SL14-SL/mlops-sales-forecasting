from pathlib import Path

import pytest

from mlops_sales_forecasting.inference.releases.cloud_exporter import (
    export_active_release,
)
from mlops_sales_forecasting.inference.releases.contracts import (
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_sales_forecasting.inference.releases.manifest import (
    write_serving_manifest,
)
from mlops_sales_forecasting.inference.releases.pointer import (
    ReleaseOperation,
    activate_release_pointer,
    load_active_release_pointer,
)
from mlops_sales_forecasting.inference.releases.repository import (
    load_release_manifest,
)
from mlops_sales_forecasting.inference.releases.storage import (
    build_artifact_reference,
    build_release_paths,
)


def prepare_source_release(
    tmp_path: Path,
) -> Path:
    models_path = tmp_path / "source-models"
    release_id = "release-source"
    paths = build_release_paths(
        models_path=str(models_path),
        release_id=release_id,
    )
    release_root = Path(paths["release_root"])
    release_root.mkdir(
        parents=True,
    )

    artifact_files = {
        "store_metadata": (release_root / "store_metadata.parquet"),
        "store_state": (release_root / "store_state.json"),
        "known_calendar": (release_root / "known_calendar.parquet"),
    }

    artifact_files["store_metadata"].write_bytes(b"store metadata")
    artifact_files["store_state"].write_text(
        '{"1":{"history":[100.0]}}',
        encoding="utf-8",
    )
    artifact_files["known_calendar"].write_bytes(b"known calendar")

    references = {
        name: build_artifact_reference(
            relative_path=path.name,
            artifact_uri=str(path),
        )
        for name, path in artifact_files.items()
    }

    manifest = ServingReleaseManifest(
        schema_version=1,
        release_id=release_id,
        created_at_utc=("2026-10-02T05:00:00+00:00"),
        task_type=TaskType.FORECASTING,
        model=ModelReference(
            name="forecast-model",
            version="2",
            run_id="run-2",
            uri=("models:/forecast-model/2"),
            model_type="xgboost",
        ),
        artifacts=references,
        metadata={
            "target_transformation": ("log1p"),
        },
    )

    write_serving_manifest(
        paths["manifest"],
        manifest,
    )
    activate_release_pointer(
        models_path=str(models_path),
        release_id=release_id,
        operation=(ReleaseOperation.BOOTSTRAP),
        updated_at_utc=("2026-10-02T05:01:00+00:00"),
    )

    return models_path


def fake_model_downloader(
    model_uri: str,
    destination: str,
) -> str:
    assert model_uri == ("models:/forecast-model/2")

    model_root = Path(destination)
    (model_root / "MLmodel").write_text(
        "flavors:\n  xgboost: {}\n",
        encoding="utf-8",
    )
    (model_root / "model.ubj").write_bytes(b"portable model")

    environment_directory = model_root / "environment"
    environment_directory.mkdir()
    (environment_directory / "python_env.yaml").write_text(
        "python: 3.12.9\n",
        encoding="utf-8",
    )

    return str(model_root)


def test_exports_active_release_with_portable_model(
    tmp_path: Path,
) -> None:
    source_models = prepare_source_release(tmp_path)
    target_models = tmp_path / "target-models"

    result = export_active_release(
        source_models_path=str(source_models),
        target_models_path=str(target_models),
        model_downloader=(fake_model_downloader),
        release_id="release-cloud",
    )

    expected_release_root = target_models / "serving_releases" / "release-cloud"

    assert result.manifest.release_id == ("release-cloud")
    assert result.release_root == str(expected_release_root)
    assert result.manifest.model.uri == str(expected_release_root / "model")
    assert result.manifest.metadata == {
        "target_transformation": ("log1p"),
        "source_release_id": ("release-source"),
        "source_model_uri": ("models:/forecast-model/2"),
    }

    assert set(result.manifest.artifacts) == {
        "store_metadata",
        "store_state",
        "known_calendar",
        "model_file_0001",
        "model_file_0002",
        "model_file_0003",
    }

    assert (expected_release_root / "store_metadata.parquet").is_file()
    assert (expected_release_root / "store_state.json").is_file()
    assert (expected_release_root / "known_calendar.parquet").is_file()
    assert (expected_release_root / "model" / "MLmodel").is_file()
    assert (expected_release_root / "model" / "model.ubj").is_file()
    assert (expected_release_root / "model" / "environment" / "python_env.yaml").is_file()

    pointer = load_active_release_pointer(models_path=str(target_models))

    assert pointer.release_id == ("release-cloud")
    assert pointer.operation is (ReleaseOperation.BOOTSTRAP)


def test_export_preserves_source_pointer(
    tmp_path: Path,
) -> None:
    source_models = prepare_source_release(tmp_path)

    export_active_release(
        source_models_path=str(source_models),
        target_models_path=str(tmp_path / "target-models"),
        model_downloader=(fake_model_downloader),
        release_id="release-cloud",
    )

    source_pointer = load_active_release_pointer(models_path=str(source_models))

    assert source_pointer.release_id == ("release-source")


def test_existing_target_release_is_rejected(
    tmp_path: Path,
) -> None:
    source_models = prepare_source_release(tmp_path)
    target_models = tmp_path / "target-models"

    export_active_release(
        source_models_path=str(source_models),
        target_models_path=str(target_models),
        model_downloader=(fake_model_downloader),
        release_id="release-cloud",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        export_active_release(
            source_models_path=str(source_models),
            target_models_path=str(target_models),
            model_downloader=(fake_model_downloader),
            release_id="release-cloud",
        )


def test_exported_manifest_can_be_loaded(
    tmp_path: Path,
) -> None:
    source_models = prepare_source_release(tmp_path)
    target_models = tmp_path / "target-models"

    export_active_release(
        source_models_path=str(source_models),
        target_models_path=str(target_models),
        model_downloader=(fake_model_downloader),
        release_id="release-cloud",
    )

    manifest, release_root = load_release_manifest(
        models_path=str(target_models),
        release_id=("release-cloud"),
    )

    assert manifest.release_id == ("release-cloud")
    assert release_root.endswith("serving_releases/release-cloud")
    assert manifest.model.uri.endswith("serving_releases/release-cloud/model")
