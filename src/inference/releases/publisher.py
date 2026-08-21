import uuid

from typing import Any
from datetime import timezone, datetime
from src.storage.filesystem import file_exists, remove_file

from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingReleaseManifest,
)

from src.inference.releases.storage import (
    copy_uri, 
    sha256_uri, 
    write_json, 
    load_json, 
    build_release_paths,
)

from src.inference.releases.repository import (
    activate_release_pointer,
    load_active_release_id
)

def build_release_id(
    model_version: str,
) -> str:
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    unique_suffix = uuid.uuid4().hex[:8]

    return (
        f"release-{timestamp}"
        f"-v{model_version}"
        f"-{unique_suffix}"
    )
     
def publish_serving_release(
    *,
    models_path: str,
    model_name: str,
    model_version: str,
    model_run_id: str,
    model_type: str,
    target_transformation: str,
    dataset_version: str | None,
    config_hash: str | None,
    git_commit: str | None,
    store_metadata_source: str,
    store_state_source: str,
    known_calendar_source: str,
    prediction_probe_payload: dict[str, Any],
) -> ServingReleaseManifest:
    """
    Publish a complete immutable serving release.

    Publication protocol:
    1. Copy artifacts into an immutable release prefix.
    2. Verify copied artifacts using SHA-256.
    3. Write the release manifest.
    4. Update the active pointer last.

    If any operation before step 4 fails, the previous active release remains
    selected.
    """
    resolved_model_version = str(
        model_version
    )

    release_id = build_release_id(
        resolved_model_version
    )

    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if file_exists(paths["manifest"]):
        raise FileExistsError(
            "Serving release already exists: "
            f"{release_id}"
        )

    artifact_sources = {
        "store_metadata": (
            store_metadata_source
        ),
        "store_state": (
            store_state_source
        ),
        "known_calendar": (
            known_calendar_source
        ),
    }

    artifact_references: dict[
        str,
        ServingArtifactReference,
    ] = {}

    probe_inputs = (
        prediction_probe_payload.get(
            "inputs"
        )
    )

    if (
        not isinstance(probe_inputs, list)
        or not probe_inputs
    ):
        raise ValueError(
            "Prediction probe payload must "
            "contain non-empty inputs."
        )

    try:
        for artifact_name, source_path in (
            artifact_sources.items()
        ):
            target_path = paths[artifact_name]

            source_hash = sha256_uri(
                source_path
            )

            copy_uri(
                source_path,
                target_path,
            )

            target_hash = sha256_uri(
                target_path
            )

            if target_hash != source_hash:
                raise ValueError(
                    "Serving artifact checksum mismatch "
                    f"after copy: {artifact_name}"
                )

            artifact_references[
                artifact_name
            ] = ServingArtifactReference(
                # Paths are relative to the release root.
                path={
                    "store_metadata": "store.parquet",
                    "store_state": "latest_state.json",
                    "known_calendar": (
                        "known_calendar.parquet"
                    ),
                }[artifact_name],
                sha256=target_hash,
            )
            write_json(
                paths["prediction_probe"],
                prediction_probe_payload,
            )

            prediction_probe_hash = sha256_uri(
                paths["prediction_probe"]
            )

            artifact_references[
                "prediction_probe"
            ] = ServingArtifactReference(
                path="prediction_probe.json",
                sha256=prediction_probe_hash,
            )


        manifest = ServingReleaseManifest(
            schema_version=2,
            release_id=release_id,
            created_at_utc=datetime.now(
                timezone.utc
            ).isoformat(),
            model_name=model_name,
            model_version=(
                resolved_model_version
            ),
            model_run_id=model_run_id,
            model_uri=(
                f"models:/{model_name}/"
                f"{resolved_model_version}"
            ),
            model_type=model_type,
            target_transformation=(
                target_transformation
            ),
            dataset_version=dataset_version,
            config_hash=config_hash,
            git_commit=git_commit,
            store_metadata=artifact_references[
                "store_metadata"
            ],
            store_state=artifact_references[
                "store_state"
            ],
            known_calendar=artifact_references[
                "known_calendar"
            ],
            prediction_probe=artifact_references[
                "prediction_probe"
            ],
        )

        # Manifest is written only after all artifacts were verified.
        write_json(
            paths["manifest"],
            manifest.to_dict(),
        )

        # Read-after-write validation before publishing the pointer.
        stored_manifest = load_json(
            paths["manifest"]
        )

        if (
            stored_manifest.get("release_id")
            != release_id
        ):
            raise ValueError(
                "Stored serving manifest failed "
                "read-after-write validation."
            )

        # Commit point: update pointer only after complete publication.
        previous_release_id = None

        if file_exists(
            paths["active_pointer"]
        ):
            previous_release_id = (
                load_active_release_id(
                    models_path=models_path,
                )
            )

        activate_release_pointer(
            models_path=models_path,
            release_id=release_id,
            operation="promotion",
            previous_release_id=(
                previous_release_id
            ),
        )

        return manifest

    except Exception:
        # The active pointer has not been changed yet.
        # Remove incomplete release objects where possible.
        for key in (
            "manifest",
            "store_metadata",
            "store_state",
            "known_calendar",
            "prediction_probe",
        ):
            remove_file(
                paths[key]
            )

        raise