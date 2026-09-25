from pathlib import Path

import pytest

from mlops_sales_forecasting.inference.releases.pointer import (
    ActiveReleasePointer,
    ReleaseOperation,
    activate_release_pointer,
    load_active_release_id,
    load_active_release_pointer,
    parse_active_release_pointer,
    validate_active_release_pointer,
)
from mlops_sales_forecasting.inference.releases.storage import (
    build_release_paths,
)

TIMESTAMP = "2026-09-14T08:00:00+00:00"


def build_valid_pointer() -> ActiveReleasePointer:
    return ActiveReleasePointer(
        schema_version=1,
        release_id="release-2",
        previous_release_id="release-1",
        operation=ReleaseOperation.ACTIVATION,
        updated_at_utc=TIMESTAMP,
    )


def create_manifest(
    models_path: Path,
    release_id: str,
) -> None:
    paths = build_release_paths(
        models_path=str(models_path),
        release_id=release_id,
    )
    manifest_path = Path(paths["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )


def test_pointer_converts_to_dictionary() -> None:
    result = build_valid_pointer().to_dict()

    assert result == {
        "schema_version": 1,
        "release_id": "release-2",
        "previous_release_id": "release-1",
        "operation": ReleaseOperation.ACTIVATION,
        "updated_at_utc": TIMESTAMP,
    }


def test_parse_active_release_pointer() -> None:
    pointer = build_valid_pointer()

    result = parse_active_release_pointer(
        pointer.to_dict()
    )

    assert result == pointer
    assert result.operation is ReleaseOperation.ACTIVATION


def test_validate_pointer_rejects_schema_version() -> None:
    pointer = ActiveReleasePointer(
        schema_version=2,
        release_id="release-2",
        previous_release_id=None,
        operation=ReleaseOperation.ACTIVATION,
        updated_at_utc=TIMESTAMP,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported active release pointer schema version",
    ):
        validate_active_release_pointer(pointer)


def test_parse_pointer_rejects_unknown_operation() -> None:
    payload = build_valid_pointer().to_dict()
    payload["operation"] = "delete"

    with pytest.raises(
        ValueError,
        match="unsupported operation",
    ):
        parse_active_release_pointer(payload)


def test_validate_pointer_rejects_timestamp_without_timezone() -> None:
    pointer = ActiveReleasePointer(
        schema_version=1,
        release_id="release-2",
        previous_release_id=None,
        operation=ReleaseOperation.ACTIVATION,
        updated_at_utc="2026-09-14T08:00:00",
    )

    with pytest.raises(
        ValueError,
        match="must include a timezone",
    ):
        validate_active_release_pointer(pointer)


def test_load_active_pointer_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Active serving release pointer not found",
    ):
        load_active_release_pointer(
            models_path=str(tmp_path)
        )


def test_activate_pointer_requires_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Cannot activate release without manifest",
    ):
        activate_release_pointer(
            models_path=str(tmp_path),
            release_id="release-1",
        )


def test_activate_and_load_release_pointer(
    tmp_path: Path,
) -> None:
    create_manifest(tmp_path, "release-1")

    result = activate_release_pointer(
        models_path=str(tmp_path),
        release_id="release-1",
        operation=ReleaseOperation.BOOTSTRAP,
        updated_at_utc=TIMESTAMP,
    )

    assert result.release_id == "release-1"
    assert result.previous_release_id is None
    assert result.operation is ReleaseOperation.BOOTSTRAP

    assert load_active_release_id(
        models_path=str(tmp_path)
    ) == "release-1"


def test_activation_records_previous_release(
    tmp_path: Path,
) -> None:
    create_manifest(tmp_path, "release-1")
    create_manifest(tmp_path, "release-2")

    activate_release_pointer(
        models_path=str(tmp_path),
        release_id="release-1",
        updated_at_utc=TIMESTAMP,
    )
    result = activate_release_pointer(
        models_path=str(tmp_path),
        release_id="release-2",
        operation=ReleaseOperation.ROLLBACK,
        updated_at_utc="2026-09-14T09:00:00+00:00",
    )

    assert result.release_id == "release-2"
    assert result.previous_release_id == "release-1"
    assert result.operation is ReleaseOperation.ROLLBACK