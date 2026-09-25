from dataclasses import replace

import pytest

from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_sales_forecasting.inference.releases.policy import (
    validate_task_manifest,
)

VALID_CHECKSUM = "d" * 64


def artifact(path: str) -> ArtifactReference:
    return ArtifactReference(
        path=path,
        sha256=VALID_CHECKSUM,
    )



def build_valid_manifest() -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="forecasting-release",
        created_at_utc="2026-09-14T08:00:00+00:00",
        task_type=TaskType.FORECASTING,
        model=ModelReference(
            name="forecasting-model",
            version="1",
            run_id="forecasting-run",
            uri="models:/forecasting-model/1",
            model_type="xgboost",
        ),
        artifacts={
            "store_metadata": artifact("store.parquet"),
            "store_state": artifact("latest_state.json"),
            "known_calendar": artifact("known_calendar.parquet"),
        },
        metadata={
            "target_transformation": "log1p",
        },
    )


def test_validate_task_manifest() -> None:
    validate_task_manifest(build_valid_manifest())


@pytest.mark.parametrize(
    "missing_artifact",
    [
        "store_metadata",
        "store_state",
        "known_calendar",
    ],
)
def test_manifest_requires_forecasting_artifacts(
    missing_artifact: str,
) -> None:
    manifest = build_valid_manifest()
    remaining_artifacts = dict(manifest.artifacts)
    del remaining_artifacts[missing_artifact]

    invalid_manifest = replace(
        manifest,
        artifacts=remaining_artifacts,
    )

    with pytest.raises(
        ValueError,
        match="missing required artifacts",
    ):
        validate_task_manifest(invalid_manifest)


def test_manifest_requires_target_transformation() -> None:
    manifest = replace(
        build_valid_manifest(),
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="requires a target_transformation",
    ):
        validate_task_manifest(manifest)


def test_manifest_rejects_classification_task() -> None:
    manifest = replace(
        build_valid_manifest(),
        task_type=TaskType.CLASSIFICATION,
    )

    with pytest.raises(
        ValueError,
        match="Expected a forecasting",
    ):
        validate_task_manifest(manifest)
