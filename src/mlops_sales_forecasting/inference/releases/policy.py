from .contracts import ServingReleaseManifest, TaskType, validate_serving_manifest

_REQUIRED_ARTIFACTS = {
    "store_metadata",
    "store_state",
    "known_calendar",
}


def validate_task_manifest(
    manifest: ServingReleaseManifest,
) -> None:
    """Validate forecasting-specific release requirements."""
    validate_serving_manifest(manifest)

    if manifest.task_type is not TaskType.FORECASTING:
        raise ValueError(
            "Expected a forecasting serving manifest."
        )

    missing_artifacts = (
        _REQUIRED_ARTIFACTS - manifest.artifacts.keys()
    )

    if missing_artifacts:
        raise ValueError(
            "Serving manifest is missing required artifacts: "
            f"{sorted(missing_artifacts)}."
        )

    metadata = manifest.metadata or {}
    target_transformation = metadata.get(
        "target_transformation"
    )

    if (
        not isinstance(target_transformation, str)
        or not target_transformation.strip()
    ):
        raise ValueError(
            "Forecasting serving manifest requires a "
            "target_transformation."
        )
