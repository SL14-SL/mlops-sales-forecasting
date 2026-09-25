from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ...configs.paths import join_uri
from ...storage.filesystem import file_exists
from .storage import (
    ACTIVE_RELEASE_FILE_NAME,
    build_release_paths,
    load_json,
    validate_release_id,
    write_json,
)

POINTER_SCHEMA_VERSION = 1


class ReleaseOperation(StrEnum):
    """Operation that changed the active serving release."""

    ACTIVATION = "activation"
    BOOTSTRAP = "bootstrap"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class ActiveReleasePointer:
    """Reference to the currently active serving release."""

    schema_version: int
    release_id: str
    previous_release_id: str | None
    operation: ReleaseOperation
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the pointer into a serializable dictionary."""
        return asdict(self)


def validate_active_release_pointer(
    pointer: ActiveReleasePointer,
) -> None:
    """Validate an active-release pointer."""
    if not isinstance(pointer, ActiveReleasePointer):
        raise ValueError(
            "Invalid active serving release pointer."
        )

    if pointer.schema_version != POINTER_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported active release pointer schema version: "
            f"{pointer.schema_version}."
        )

    validate_release_id(pointer.release_id)

    if pointer.previous_release_id is not None:
        validate_release_id(pointer.previous_release_id)

    if not isinstance(pointer.operation, ReleaseOperation):
        raise ValueError(
            "Active release pointer has an invalid operation."
        )

    try:
        timestamp = datetime.fromisoformat(
            pointer.updated_at_utc.replace("Z", "+00:00")
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "Active release pointer has an invalid timestamp."
        ) from exc

    if timestamp.tzinfo is None:
        raise ValueError(
            "Active release pointer timestamp must include a timezone."
        )


def parse_active_release_pointer(
    payload: object,
) -> ActiveReleasePointer:
    """Parse and validate an active-release pointer."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Active release pointer must contain a JSON object."
        )

    required_fields = {
        "schema_version",
        "release_id",
        "operation",
        "updated_at_utc",
    }
    missing_fields = required_fields - payload.keys()

    if missing_fields:
        raise ValueError(
            "Active release pointer is missing required fields: "
            f"{sorted(missing_fields)}."
        )

    try:
        operation = ReleaseOperation(str(payload["operation"]))
    except ValueError as exc:
        raise ValueError(
            "Active release pointer has an unsupported operation."
        ) from exc

    previous_release_id = payload.get("previous_release_id")

    pointer = ActiveReleasePointer(
        schema_version=int(payload["schema_version"]),
        release_id=str(payload["release_id"]),
        previous_release_id=(
            str(previous_release_id)
            if previous_release_id is not None
            else None
        ),
        operation=operation,
        updated_at_utc=str(payload["updated_at_utc"]),
    )

    validate_active_release_pointer(pointer)
    return pointer


def active_pointer_path(models_path: str) -> str:
    """Return the active-release pointer location."""
    return join_uri(
        models_path,
        ACTIVE_RELEASE_FILE_NAME,
    )


def load_active_release_pointer(
    *,
    models_path: str,
) -> ActiveReleasePointer:
    """Load and validate the active-release pointer."""
    pointer_path = active_pointer_path(models_path)

    if not file_exists(pointer_path):
        raise FileNotFoundError(
            "Active serving release pointer not found: "
            f"{pointer_path}"
        )

    return parse_active_release_pointer(
        load_json(pointer_path)
    )


def load_active_release_id(
    *,
    models_path: str,
) -> str:
    """Return the active serving-release identifier."""
    return load_active_release_pointer(
        models_path=models_path
    ).release_id


def activate_release_pointer(
    *,
    models_path: str,
    release_id: str,
    operation: ReleaseOperation = ReleaseOperation.ACTIVATION,
    previous_release_id: str | None = None,
    updated_at_utc: str | None = None,
) -> ActiveReleasePointer:
    """Activate a validated serving release."""
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if not file_exists(paths["manifest"]):
        raise FileNotFoundError(
            "Cannot activate release without manifest: "
            f"{paths['manifest']}"
        )

    pointer_path = paths["active_pointer"]

    if (
        previous_release_id is None
        and file_exists(pointer_path)
    ):
        current_release_id = load_active_release_id(
            models_path=models_path
        )

        if current_release_id != release_id:
            previous_release_id = current_release_id

    pointer = ActiveReleasePointer(
        schema_version=POINTER_SCHEMA_VERSION,
        release_id=release_id,
        previous_release_id=previous_release_id,
        operation=operation,
        updated_at_utc=(
            updated_at_utc
            or datetime.now(UTC).isoformat()
        ),
    )
    validate_active_release_pointer(pointer)

    write_json(
        pointer_path,
        pointer.to_dict(),
    )

    stored_pointer = load_active_release_pointer(
        models_path=models_path
    )

    if stored_pointer != pointer:
        raise RuntimeError(
            "Active release pointer verification failed."
        )

    return stored_pointer