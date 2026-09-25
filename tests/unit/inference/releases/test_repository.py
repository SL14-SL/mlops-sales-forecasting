from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_sales_forecasting.inference.releases.manifest import (
    write_serving_manifest,
)
from mlops_sales_forecasting.inference.releases.pointer import (
    ReleaseOperation,
)
from mlops_sales_forecasting.inference.releases.repository import (
    activate_release,
    list_release_manifests,
    load_active_release_manifest,
    load_release_manifest,
    rollback_active_release,
)
from mlops_sales_forecasting.inference.releases.storage import (
    build_release_paths,
)

VALID_CHECKSUM = "c" * 64


def create_release(
    models_path: Path,
    release_id: str,
    *,
    created_at: datetime,
    task_type: TaskType = TaskType.CLASSIFICATION,
) -> ServingReleaseManifest:
    manifest = ServingReleaseManifest(
        schema_version=1,
        release_id=release_id,
        created_at_utc=created_at.isoformat(),
        task_type=task_type,
        model=ModelReference(
            name="example-model",
            version=release_id.removeprefix("release-"),
            run_id=f"run-{release_id}",
            uri=f"models:/example-model/{release_id}",
            model_type="xgboost",
        ),
        artifacts={
            "feature_schema": ArtifactReference(
                path="feature_schema.json",
                sha256=VALID_CHECKSUM,
            ),
        },
    )

    paths = build_release_paths(
        models_path=str(models_path),
        release_id=release_id,
    )
    write_serving_manifest(
        paths["manifest"],
        manifest,
    )
    return manifest


def test_load_release_manifest(tmp_path: Path) -> None:
    expected = create_release(
        tmp_path,
        "release-1",
        created_at=datetime(2026, 9, 14, tzinfo=UTC),
    )

    manifest, release_root = load_release_manifest(
        models_path=str(tmp_path),
        release_id="release-1",
    )

    assert manifest == expected
    assert release_root.endswith(
        "serving_releases/release-1"
    )


def test_load_release_manifest_rejects_missing_release(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Serving release manifest not found",
    ):
        load_release_manifest(
            models_path=str(tmp_path),
            release_id="release-404",
        )


def test_list_release_manifests_returns_newest_first(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 9, 14, tzinfo=UTC)

    create_release(
        tmp_path,
        "release-1",
        created_at=start,
    )
    create_release(
        tmp_path,
        "release-3",
        created_at=start + timedelta(hours=2),
    )
    create_release(
        tmp_path,
        "release-2",
        created_at=start + timedelta(hours=1),
    )

    result = list_release_manifests(
        models_path=str(tmp_path)
    )

    assert [
        manifest.release_id
        for manifest in result
    ] == [
        "release-3",
        "release-2",
        "release-1",
    ]


def test_list_release_manifests_returns_empty_list(
    tmp_path: Path,
) -> None:
    assert list_release_manifests(
        models_path=str(tmp_path)
    ) == []


def test_activate_and_load_release(
    tmp_path: Path,
) -> None:
    expected = create_release(
        tmp_path,
        "release-1",
        created_at=datetime(2026, 9, 14, tzinfo=UTC),
    )

    pointer = activate_release(
        models_path=str(tmp_path),
        release_id="release-1",
        operation=ReleaseOperation.BOOTSTRAP,
        expected_task_type=TaskType.CLASSIFICATION,
        updated_at_utc="2026-09-14T08:00:00+00:00",
    )
    manifest, _ = load_active_release_manifest(
        models_path=str(tmp_path)
    )

    assert pointer.release_id == "release-1"
    assert manifest == expected


def test_activate_release_rejects_wrong_task_type(
    tmp_path: Path,
) -> None:
    create_release(
        tmp_path,
        "release-1",
        created_at=datetime(2026, 9, 14, tzinfo=UTC),
        task_type=TaskType.FORECASTING,
    )

    with pytest.raises(
        ValueError,
        match="task type does not match project",
    ):
        activate_release(
            models_path=str(tmp_path),
            release_id="release-1",
            expected_task_type=TaskType.CLASSIFICATION,
        )


def test_rollback_activates_previous_release(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 9, 14, tzinfo=UTC)

    create_release(
        tmp_path,
        "release-1",
        created_at=start,
    )
    create_release(
        tmp_path,
        "release-2",
        created_at=start + timedelta(hours=1),
    )

    activate_release(
        models_path=str(tmp_path),
        release_id="release-1",
        updated_at_utc="2026-09-14T08:00:00+00:00",
    )
    activate_release(
        models_path=str(tmp_path),
        release_id="release-2",
        updated_at_utc="2026-09-14T09:00:00+00:00",
    )

    result = rollback_active_release(
        models_path=str(tmp_path),
        expected_task_type=TaskType.CLASSIFICATION,
        updated_at_utc="2026-09-14T10:00:00+00:00",
    )

    assert result.release_id == "release-1"
    assert result.previous_release_id == "release-2"
    assert result.operation is ReleaseOperation.ROLLBACK


def test_rollback_rejects_missing_candidate(
    tmp_path: Path,
) -> None:
    create_release(
        tmp_path,
        "release-1",
        created_at=datetime(2026, 9, 14, tzinfo=UTC),
    )
    activate_release(
        models_path=str(tmp_path),
        release_id="release-1",
        updated_at_utc="2026-09-14T08:00:00+00:00",
    )

    with pytest.raises(
        ValueError,
        match="contains no rollback candidate",
    ):
        rollback_active_release(
            models_path=str(tmp_path)
        )