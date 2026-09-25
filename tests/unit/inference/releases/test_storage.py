import hashlib
import io
from pathlib import Path

import pytest

from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
)
from mlops_sales_forecasting.inference.releases.storage import (
    ACTIVE_RELEASE_FILE_NAME,
    COPY_CHUNK_SIZE,
    MANIFEST_FILE_NAME,
    build_artifact_reference,
    build_release_paths,
    copy_uri,
    load_json,
    resolve_artifact_uri,
    sha256_uri,
    write_json,
)


def test_copy_uri_copies_local_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "nested" / "target.bin"
    source.write_bytes(b"example content")

    copy_uri(str(source), str(target))

    assert target.read_bytes() == b"example content"


def test_copy_uri_rejects_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "missing.bin"
    target = tmp_path / "target.bin"

    with pytest.raises(
        FileNotFoundError,
        match="Serving release source not found",
    ):
        copy_uri(str(source), str(target))


def test_copy_uri_handles_multiple_chunks(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    target = tmp_path / "copied.bin"
    content = b"x" * (COPY_CHUNK_SIZE + 100)
    source.write_bytes(content)

    copy_uri(str(source), str(target))

    assert target.read_bytes() == content


def test_sha256_uri_returns_expected_checksum(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    content = b"immutable model artifact"
    artifact.write_bytes(content)

    result = sha256_uri(str(artifact))

    assert result == hashlib.sha256(content).hexdigest()


def test_write_and_load_json_round_trip(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "nested" / "payload.json"
    payload = {
        "release_id": "release-1",
        "enabled": True,
    }

    write_json(str(json_path), payload)
    result = load_json(str(json_path))

    assert result == payload
    assert json_path.read_text(encoding="utf-8").endswith("\n")


def test_load_json_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "invalid.json"
    json_path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON document",
    ):
        load_json(str(json_path))


def test_load_json_rejects_non_object(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "list.json"
    json_path.write_text(
        '["first", "second"]',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Expected JSON object",
    ):
        load_json(str(json_path))


def test_build_release_paths_returns_common_paths() -> None:
    result = build_release_paths(
        models_path="artifacts/models",
        release_id="release-123",
    )

    assert result == {
        "release_root": (
            "artifacts/models/serving_releases/release-123"
        ),
        "manifest": (
            "artifacts/models/serving_releases/"
            f"release-123/{MANIFEST_FILE_NAME}"
        ),
        "active_pointer": (
            f"artifacts/models/{ACTIVE_RELEASE_FILE_NAME}"
        ),
    }


def test_build_release_paths_adds_dynamic_artifacts() -> None:
    result = build_release_paths(
        models_path="gs://example-bucket/models",
        release_id="release-123",
        artifact_filenames={
            "feature_schema": "feature_schema.json",
            "probe": "probes/prediction.json",
        },
    )

    assert result["feature_schema"] == (
        "gs://example-bucket/models/serving_releases/"
        "release-123/feature_schema.json"
    )
    assert result["probe"] == (
        "gs://example-bucket/models/serving_releases/"
        "release-123/probes/prediction.json"
    )


def test_build_release_paths_rejects_invalid_release_id() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid serving release ID",
    ):
        build_release_paths(
            models_path="artifacts/models",
            release_id="../release",
        )


def test_build_release_paths_rejects_unsafe_artifact_path() -> None:
    with pytest.raises(
        ValueError,
        match="must be relative",
    ):
        build_release_paths(
            models_path="artifacts/models",
            release_id="release-123",
            artifact_filenames={
                "schema": "../feature_schema.json",
            },
        )


def test_build_release_paths_rejects_reserved_name() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate release path name",
    ):
        build_release_paths(
            models_path="artifacts/models",
            release_id="release-123",
            artifact_filenames={
                "manifest": "other-manifest.json",
            },
        )


def test_build_artifact_reference(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "feature_schema.json"
    content = b'{"columns": ["feature"]}'
    artifact.write_bytes(content)

    result = build_artifact_reference(
        relative_path="feature_schema.json",
        artifact_uri=str(artifact),
    )

    assert result == ArtifactReference(
        path="feature_schema.json",
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_build_artifact_reference_rejects_missing_file(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Serving artifact not found",
    ):
        build_artifact_reference(
            relative_path="missing.json",
            artifact_uri=str(artifact),
        )


def test_resolve_artifact_uri_validates_checksum(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    artifact = release_root / "feature_schema.json"
    artifact.parent.mkdir(parents=True)
    content = b'{"columns": ["feature"]}'
    artifact.write_bytes(content)

    reference = ArtifactReference(
        path="feature_schema.json",
        sha256=hashlib.sha256(content).hexdigest(),
    )

    result = resolve_artifact_uri(
        release_root=str(release_root),
        reference=reference,
    )

    assert result == str(artifact)


def test_resolve_artifact_uri_rejects_checksum_mismatch(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    artifact = release_root / "feature_schema.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"modified content")

    reference = ArtifactReference(
        path="feature_schema.json",
        sha256="a" * 64,
    )

    with pytest.raises(
        ValueError,
        match="checksum mismatch",
    ):
        resolve_artifact_uri(
            release_root=str(release_root),
            reference=reference,
        )


def test_resolve_artifact_uri_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    reference = ArtifactReference(
        path="missing.json",
        sha256="a" * 64,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Serving artifact not found",
    ):
        resolve_artifact_uri(
            release_root=str(tmp_path),
            reference=reference,
        )


def test_copy_stream_supports_binary_streams() -> None:
    from mlops_sales_forecasting.inference.releases.storage import (
        _copy_stream,
    )

    source = io.BytesIO(b"stream content")
    target = io.BytesIO()

    _copy_stream(source, target)

    assert target.getvalue() == b"stream content"