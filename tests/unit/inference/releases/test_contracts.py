from dataclasses import replace

import pytest

from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
    validate_artifact_reference,
    validate_model_reference,
    validate_serving_manifest,
)

VALID_CHECKSUM = "a" * 64


def build_valid_manifest() -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-20260914",
        created_at_utc="2026-09-14T08:00:00Z",
        task_type=TaskType.CLASSIFICATION,
        model=ModelReference(
            name="example-model-dev",
            version="3",
            run_id="run-3",
            uri="models:/example-model-dev@champion",
            model_type="xgboost",
        ),
        artifacts={
            "feature_schema": ArtifactReference(
                path="feature_schema.json",
                sha256=VALID_CHECKSUM,
            ),
        },
        dataset_version="dataset-v1",
        config_hash="config-hash",
        git_commit="abc123",
        metadata={
            "decision_threshold": 0.42,
        },
    )


def test_manifest_converts_to_dictionary() -> None:
    manifest = build_valid_manifest()

    result = manifest.to_dict()

    assert result["release_id"] == "release-20260914"
    assert result["task_type"] == TaskType.CLASSIFICATION
    assert result["model"]["version"] == "3"
    assert (
        result["artifacts"]["feature_schema"]["sha256"]
        == VALID_CHECKSUM
    )


def test_validate_serving_manifest_accepts_valid_manifest() -> None:
    validate_serving_manifest(build_valid_manifest())


def test_validate_serving_manifest_accepts_forecasting_task() -> None:
    manifest = replace(
        build_valid_manifest(),
        task_type=TaskType.FORECASTING,
        artifacts={
            "store_state": ArtifactReference(
                path="store_state.json",
                sha256=VALID_CHECKSUM,
            ),
        },
        metadata={
            "target_transformation": "log1p",
        },
    )

    validate_serving_manifest(manifest)


def test_validate_serving_manifest_rejects_schema_version() -> None:
    manifest = replace(
        build_valid_manifest(),
        schema_version=2,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported serving manifest schema version",
    ):
        validate_serving_manifest(manifest)


def test_validate_serving_manifest_rejects_empty_release_id() -> None:
    manifest = replace(
        build_valid_manifest(),
        release_id="",
    )

    with pytest.raises(
        ValueError,
        match="invalid release ID",
    ):
        validate_serving_manifest(manifest)


def test_validate_serving_manifest_rejects_invalid_timestamp() -> None:
    manifest = replace(
        build_valid_manifest(),
        created_at_utc="not-a-timestamp",
    )

    with pytest.raises(
        ValueError,
        match="invalid creation timestamp",
    ):
        validate_serving_manifest(manifest)


def test_validate_serving_manifest_requires_timestamp_timezone() -> None:
    manifest = replace(
        build_valid_manifest(),
        created_at_utc="2026-09-14T08:00:00",
    )

    with pytest.raises(
        ValueError,
        match="must include a timezone",
    ):
        validate_serving_manifest(manifest)


def test_validate_serving_manifest_rejects_empty_artifacts() -> None:
    manifest = replace(
        build_valid_manifest(),
        artifacts={},
    )

    with pytest.raises(
        ValueError,
        match="contains no artifacts",
    ):
        validate_serving_manifest(manifest)


def test_validate_artifact_reference_rejects_absolute_path() -> None:
    reference = ArtifactReference(
        path="/tmp/feature_schema.json",
        sha256=VALID_CHECKSUM,
    )

    with pytest.raises(
        ValueError,
        match="path must be relative",
    ):
        validate_artifact_reference(
            reference,
            name="feature_schema",
        )


def test_validate_artifact_reference_rejects_parent_traversal() -> None:
    reference = ArtifactReference(
        path="../feature_schema.json",
        sha256=VALID_CHECKSUM,
    )

    with pytest.raises(
        ValueError,
        match="path must be relative",
    ):
        validate_artifact_reference(
            reference,
            name="feature_schema",
        )


def test_validate_artifact_reference_rejects_invalid_checksum() -> None:
    reference = ArtifactReference(
        path="feature_schema.json",
        sha256="not-a-checksum",
    )

    with pytest.raises(
        ValueError,
        match="invalid SHA-256 checksum",
    ):
        validate_artifact_reference(
            reference,
            name="feature_schema",
        )


def test_validate_model_reference_rejects_empty_model_name() -> None:
    manifest = build_valid_manifest()
    invalid_model = replace(
        manifest.model,
        name="",
    )

    with pytest.raises(
        ValueError,
        match="invalid model name",
    ):
        validate_model_reference(invalid_model)


def test_validate_serving_manifest_rejects_invalid_metadata() -> None:
    manifest = replace(
        build_valid_manifest(),
        metadata="invalid",  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="metadata must be a mapping",
    ):
        validate_serving_manifest(manifest)