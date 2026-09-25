import hashlib
from pathlib import Path

import pytest

from mlops_sales_forecasting.inference.releases.artifact_publisher import (
    PublishedServingArtifacts,
    ServingArtifactSource,
    publish_serving_artifacts,
)


def test_publishes_and_checksums_artifacts(
    tmp_path: Path,
) -> None:
    sources_root = tmp_path / "sources"
    sources_root.mkdir()

    schema_source = (
        sources_root / "feature_schema.json"
    )
    probe_source = (
        sources_root / "prediction_probe.json"
    )

    schema_content = (
        b'{"columns": ["feature"]}'
    )
    probe_content = (
        b'{"inputs": [{"feature": 1.0}]}'
    )

    schema_source.write_bytes(
        schema_content
    )
    probe_source.write_bytes(
        probe_content
    )

    models_path = tmp_path / "models"

    result = publish_serving_artifacts(
        models_path=str(models_path),
        release_id="release-7",
        sources={
            "feature_schema": (
                ServingArtifactSource(
                    source_uri=str(
                        schema_source
                    ),
                    relative_path=(
                        "feature_schema.json"
                    ),
                )
            ),
            "prediction_probe": (
                ServingArtifactSource(
                    source_uri=str(
                        probe_source
                    ),
                    relative_path=(
                        "probes/prediction.json"
                    ),
                )
            ),
        },
    )

    assert isinstance(
        result,
        PublishedServingArtifacts,
    )
    assert result.release_root == str(
        models_path
        / "serving_releases"
        / "release-7"
    )

    copied_schema = (
        Path(result.release_root)
        / "feature_schema.json"
    )
    copied_probe = (
        Path(result.release_root)
        / "probes"
        / "prediction.json"
    )

    assert (
        copied_schema.read_bytes()
        == schema_content
    )
    assert (
        copied_probe.read_bytes()
        == probe_content
    )

    assert (
        result.references[
            "feature_schema"
        ].path
        == "feature_schema.json"
    )
    assert (
        result.references[
            "feature_schema"
        ].sha256
        == hashlib.sha256(
            schema_content
        ).hexdigest()
    )
    assert (
        result.references[
            "prediction_probe"
        ].path
        == "probes/prediction.json"
    )


def test_rejects_empty_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least one artifact",
    ):
        publish_serving_artifacts(
            models_path=str(
                tmp_path / "models"
            ),
            release_id="release-1",
            sources={},
        )


def test_rejects_duplicate_destinations(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        "{}",
        encoding="utf-8",
    )
    second.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="destinations must be unique",
    ):
        publish_serving_artifacts(
            models_path=str(
                tmp_path / "models"
            ),
            release_id="release-1",
            sources={
                "first": ServingArtifactSource(
                    source_uri=str(first),
                    relative_path="shared.json",
                ),
                "second": ServingArtifactSource(
                    source_uri=str(second),
                    relative_path="shared.json",
                ),
            },
        )


def test_removes_partial_release_after_failure(
    tmp_path: Path,
) -> None:
    existing_source = (
        tmp_path / "existing.json"
    )
    existing_source.write_text(
        '{"valid": true}',
        encoding="utf-8",
    )

    missing_source = (
        tmp_path / "missing.json"
    )
    models_path = tmp_path / "models"

    with pytest.raises(
        FileNotFoundError,
        match="source not found",
    ):
        publish_serving_artifacts(
            models_path=str(models_path),
            release_id="release-1",
            sources={
                "existing": (
                    ServingArtifactSource(
                        source_uri=str(
                            existing_source
                        ),
                        relative_path=(
                            "existing.json"
                        ),
                    )
                ),
                "missing": (
                    ServingArtifactSource(
                        source_uri=str(
                            missing_source
                        ),
                        relative_path=(
                            "missing.json"
                        ),
                    )
                ),
            },
        )

    copied_existing = (
        models_path
        / "serving_releases"
        / "release-1"
        / "existing.json"
    )

    assert not copied_existing.exists()


def test_rejects_unsafe_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be relative",
    ):
        publish_serving_artifacts(
            models_path=str(
                tmp_path / "models"
            ),
            release_id="release-1",
            sources={
                "schema": (
                    ServingArtifactSource(
                        source_uri=str(source),
                        relative_path=(
                            "../schema.json"
                        ),
                    )
                ),
            },
        )


def test_rejects_source_equal_to_destination(
    tmp_path: Path,
) -> None:
    models_path = tmp_path / "models"
    target = (
        models_path
        / "serving_releases"
        / "release-1"
        / "schema.json"
    )
    target.parent.mkdir(
        parents=True
    )
    target.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source and destination must differ",
    ):
        publish_serving_artifacts(
            models_path=str(models_path),
            release_id="release-1",
            sources={
                "schema": (
                    ServingArtifactSource(
                        source_uri=str(target),
                        relative_path=(
                            "schema.json"
                        ),
                    )
                ),
            },
        )


def test_source_requires_non_empty_uri() -> None:
    with pytest.raises(
        ValueError,
        match="source URI",
    ):
        ServingArtifactSource(
            source_uri="",
            relative_path="schema.json",
        )