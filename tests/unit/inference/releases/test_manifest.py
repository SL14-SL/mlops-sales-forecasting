import json
from pathlib import Path

import pytest

from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_sales_forecasting.inference.releases.manifest import (
    manifest_to_json,
    parse_artifact_reference,
    parse_serving_manifest,
    read_serving_manifest,
    write_serving_manifest,
)

VALID_CHECKSUM = "b" * 64


def build_valid_manifest() -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-20260914",
        created_at_utc="2026-09-14T08:00:00Z",
        task_type=TaskType.CLASSIFICATION,
        model=ModelReference(
            name="example-model-dev",
            version="5",
            run_id="run-5",
            uri="models:/example-model-dev@champion",
            model_type="xgboost",
        ),
        artifacts={
            "feature_schema": ArtifactReference(
                path="feature_schema.json",
                sha256=VALID_CHECKSUM,
            ),
            "prediction_probe": ArtifactReference(
                path="prediction_probe.json",
                sha256=VALID_CHECKSUM,
            ),
        },
        dataset_version="dataset-v2",
        config_hash="config-hash",
        git_commit="abc123",
        metadata={
            "decision_threshold": 0.42,
        },
    )


def test_manifest_to_json_returns_formatted_json() -> None:
    result = manifest_to_json(build_valid_manifest())

    payload = json.loads(result)

    assert result.endswith("\n")
    assert payload["release_id"] == "release-20260914"
    assert payload["task_type"] == "classification"
    assert payload["model"]["version"] == "5"


def test_parse_serving_manifest_reconstructs_manifest() -> None:
    original_manifest = build_valid_manifest()

    result = parse_serving_manifest(
        original_manifest.to_dict()
    )

    assert result == original_manifest
    assert result.task_type is TaskType.CLASSIFICATION
    assert isinstance(
        result.model,
        ModelReference,
    )
    assert isinstance(
        result.artifacts["feature_schema"],
        ArtifactReference,
    )


def test_manifest_file_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "release" / "manifest.json"
    original_manifest = build_valid_manifest()

    write_serving_manifest(
        manifest_path,
        original_manifest,
    )
    result = read_serving_manifest(manifest_path)

    assert result == original_manifest


def test_read_serving_manifest_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="contains invalid JSON",
    ):
        read_serving_manifest(manifest_path)


def test_parse_serving_manifest_rejects_non_object() -> None:
    with pytest.raises(
        ValueError,
        match="must contain a JSON object",
    ):
        parse_serving_manifest(
            ["invalid", "manifest"]
        )


def test_parse_serving_manifest_rejects_missing_field() -> None:
    payload = build_valid_manifest().to_dict()
    del payload["release_id"]

    with pytest.raises(
        ValueError,
        match="missing required field: release_id",
    ):
        parse_serving_manifest(payload)


def test_parse_serving_manifest_rejects_unknown_task() -> None:
    payload = build_valid_manifest().to_dict()
    payload["task_type"] = "computer_vision"

    with pytest.raises(
        ValueError,
        match="unsupported task type",
    ):
        parse_serving_manifest(payload)


def test_parse_serving_manifest_rejects_invalid_artifacts() -> None:
    payload = build_valid_manifest().to_dict()
    payload["artifacts"] = ["feature_schema.json"]

    with pytest.raises(
        ValueError,
        match="artifacts must contain a JSON object",
    ):
        parse_serving_manifest(payload)


def test_parse_artifact_reference_rejects_non_object() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid serving artifact reference",
    ):
        parse_artifact_reference(
            "feature_schema.json",
            name="feature_schema",
        )


def test_parse_serving_manifest_validates_checksum() -> None:
    payload = build_valid_manifest().to_dict()
    payload["artifacts"]["feature_schema"]["sha256"] = "invalid"

    with pytest.raises(
        ValueError,
        match="invalid SHA-256 checksum",
    ):
        parse_serving_manifest(payload)