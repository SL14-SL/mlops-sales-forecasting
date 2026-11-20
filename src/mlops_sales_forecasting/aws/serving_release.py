import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..inference.bundle_loader import load_serving_bundle
from ..inference.model_loader import load_xgboost_model
from ..inference.releases.artifact_publisher import (
    ServingArtifactSource,
    publish_serving_artifacts,
)
from ..inference.releases.contracts import (
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from ..inference.releases.manifest import write_serving_manifest
from ..inference.releases.storage import (
    build_release_paths,
    resolve_artifact_uri,
)
from ..inference.serving_bundle import ServingBundle


def prepare_local_serving_release(
    *,
    candidate_directory: Path,
    models_directory: Path,
    model_name: str,
) -> ServingBundle:
    """Create and validate a local release without activating it."""
    candidate_directory = candidate_directory.resolve()
    models_directory = models_directory.resolve()

    if not model_name.strip():
        raise ValueError("model_name must not be empty.")

    if (
        candidate_directory == models_directory
        or candidate_directory.is_relative_to(models_directory)
        or models_directory.is_relative_to(candidate_directory)
    ):
        raise ValueError("Candidate and serving directories must not overlap.")

    report = json.loads(
        (candidate_directory / "output/evaluation.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (candidate_directory / "output/resolved_config.json").read_text(encoding="utf-8")
    )

    if report.get("evaluation_approved") is not True:
        raise ValueError("Candidate evaluation was not approved.")

    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Candidate report must contain a training run ID.")

    if config.get("model", {}).get("type") != "xgboost":
        raise ValueError("Serving export requires an XGBoost candidate.")

    transformation = config.get("training", {}).get("target_transformation", "none")
    if transformation not in {"none", "log1p"}:
        raise ValueError("Unsupported candidate target transformation.")

    model_root = candidate_directory / "model/model"
    if not (model_root / "MLmodel").is_file():
        raise FileNotFoundError("Candidate MLflow model is missing.")

    sources = {
        "store_metadata": ServingArtifactSource(
            source_uri=str(candidate_directory / "model/store_metadata.parquet"),
            relative_path="store_metadata.parquet",
        ),
        "store_state": ServingArtifactSource(
            source_uri=str(candidate_directory / "model/latest_state.json"),
            relative_path="store_state.json",
        ),
        "known_calendar": ServingArtifactSource(
            source_uri=str(candidate_directory / "model/known_calendar.parquet"),
            relative_path="known_calendar.parquet",
        ),
    }

    for index, path in enumerate(sorted(path for path in model_root.rglob("*") if path.is_file())):
        sources[f"model_file_{index:04d}"] = ServingArtifactSource(
            source_uri=str(path),
            relative_path=f"model/{path.relative_to(model_root).as_posix()}",
        )

    for source in sources.values():
        if not Path(source.source_uri).is_file():
            raise FileNotFoundError(f"Candidate artifact missing: {source.source_uri}")

    release_id = f"release-{uuid4().hex}"
    paths = build_release_paths(
        models_path=str(models_directory),
        release_id=release_id,
    )
    Path(paths["release_root"]).mkdir(parents=True, exist_ok=False)

    published = publish_serving_artifacts(
        models_path=str(models_directory),
        release_id=release_id,
        sources=sources,
    )

    manifest = ServingReleaseManifest(
        schema_version=1,
        release_id=release_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        task_type=TaskType.FORECASTING,
        model=ModelReference(
            name=model_name,
            version=release_id,
            run_id=run_id,
            uri=str(Path(published.release_root) / "model"),
            model_type="xgboost",
        ),
        artifacts=published.references,
        metadata={
            "target_transformation": transformation,
            "forecasting_state_schema_version": 1,
            "evaluation_metrics": report["evaluation_metrics"],
            "model_reference_source": "exported_candidate",
            "registry_registered": False,
            "purpose": "serving_smoke_test",
        },
    )

    write_serving_manifest(paths["manifest"], manifest)

    # Verify every copied file, including the model files.
    for reference in manifest.artifacts.values():
        resolve_artifact_uri(
            release_root=published.release_root,
            reference=reference,
        )

    return load_serving_bundle(
        models_path=str(models_directory),
        release_id=release_id,
        serving_alias="aws-demo",
        model_loader=load_xgboost_model,
    )
