from pathlib import Path

import pytest

from mlops_sales_forecasting.inference.releases.artifact_publisher import (
    ServingArtifactSource,
)
from mlops_sales_forecasting.inference.releases.contracts import (
    TaskType,
)
from mlops_sales_forecasting.inference.releases.publisher import (
    publish_serving_release,
)
from mlops_sales_forecasting.inference.releases.repository import (
    load_active_release_manifest,
)
from mlops_sales_forecasting.tracking.aliases import (
    AliasAssignment,
    ModelAlias,
)
from mlops_sales_forecasting.tracking.promotion import (
    PromotionDecision,
)
from mlops_sales_forecasting.tracking.promotion_service import (
    PromotionOutcome,
)
from mlops_sales_forecasting.tracking.registry import (
    ModelRegistrationResult,
)


def build_registration(
    version: str = "7",
) -> ModelRegistrationResult:
    return ModelRegistrationResult(
        registered=True,
        run_id=f"run-{version}",
        model_name="example-model",
        model_version=version,
        model_uri=(
            f"models:/example-model/{version}"
        ),
    )


def build_promotion(
    version: str = "7",
    *,
    promote: bool = True,
) -> PromotionOutcome:
    return PromotionOutcome(
        decision=PromotionDecision(
            promote=promote,
            metric_name="score",
            candidate_value=0.90,
            champion_value=0.80,
            improvement=0.10,
            reason="Candidate satisfies threshold.",
        ),
        previous_champion_version="6",
        champion_assignment=(
            AliasAssignment(
                model_name="example-model",
                model_version=version,
                alias=ModelAlias.CHAMPION,
            )
            if promote
            else None
        ),
    )



TASK_TYPE = TaskType.FORECASTING
METADATA = {
    "target_transformation": "log1p",
}


def build_sources(
    root: Path,
) -> dict[str, ServingArtifactSource]:
    store_metadata = (
        root / "store_metadata.parquet"
    )
    store_state = root / "store_state.json"
    known_calendar = (
        root / "known_calendar.parquet"
    )

    store_metadata.write_bytes(
        b"store metadata"
    )
    store_state.write_text(
        '{"stores": {}}',
        encoding="utf-8",
    )
    known_calendar.write_bytes(
        b"known calendar"
    )

    return {
        "store_metadata": (
            ServingArtifactSource(
                source_uri=str(
                    store_metadata
                ),
                relative_path=(
                    "store_metadata.parquet"
                ),
            )
        ),
        "store_state": (
            ServingArtifactSource(
                source_uri=str(store_state),
                relative_path=(
                    "store_state.json"
                ),
            )
        ),
        "known_calendar": (
            ServingArtifactSource(
                source_uri=str(
                    known_calendar
                ),
                relative_path=(
                    "known_calendar.parquet"
                ),
            )
        ),
    }



def test_publishes_and_activates_release(
    tmp_path: Path,
) -> None:
    models_path = tmp_path / "models"

    result = publish_serving_release(
        models_path=str(models_path),
        registration=build_registration(),
        promotion=build_promotion(),
        task_type=TASK_TYPE,
        model_type="xgboost",
        sources=build_sources(tmp_path),
        metadata=METADATA,
        release_id="release-7",
    )

    assert (
        result.manifest.release_id
        == "release-7"
    )
    assert (
        result.manifest.model.version
        == "7"
    )
    assert (
        result.active_pointer.release_id
        == "release-7"
    )
    assert (
        result.active_pointer.previous_release_id
        is None
    )

    active_manifest, release_root = (
        load_active_release_manifest(
            models_path=str(models_path)
        )
    )

    assert active_manifest == result.manifest
    assert (
        release_root
        == result.release_root
    )


def test_activation_tracks_previous_release(
    tmp_path: Path,
) -> None:
    models_path = tmp_path / "models"

    publish_serving_release(
        models_path=str(models_path),
        registration=build_registration("6"),
        promotion=build_promotion("6"),
        task_type=TASK_TYPE,
        model_type="xgboost",
        sources=build_sources(tmp_path),
        metadata=METADATA,
        release_id="release-6",
    )

    result = publish_serving_release(
        models_path=str(models_path),
        registration=build_registration("7"),
        promotion=build_promotion("7"),
        task_type=TASK_TYPE,
        model_type="xgboost",
        sources=build_sources(tmp_path),
        metadata=METADATA,
        release_id="release-7",
    )

    assert (
        result.active_pointer.release_id
        == "release-7"
    )
    assert (
        result.active_pointer.previous_release_id
        == "release-6"
    )


def test_rejected_promotion_creates_no_release(
    tmp_path: Path,
) -> None:
    models_path = tmp_path / "models"

    with pytest.raises(
        ValueError,
        match="successful champion promotion",
    ):
        publish_serving_release(
            models_path=str(models_path),
            registration=build_registration(),
            promotion=build_promotion(
                promote=False
            ),
            task_type=TASK_TYPE,
            model_type="xgboost",
            sources=build_sources(tmp_path),
            metadata=METADATA,
            release_id="release-rejected",
        )

    assert not (
        models_path
        / "serving_releases"
        / "release-rejected"
    ).exists()


def test_missing_source_creates_no_release(
    tmp_path: Path,
) -> None:
    models_path = tmp_path / "models"
    sources = build_sources(tmp_path)

    first_source = next(
        iter(sources.values())
    )
    Path(first_source.source_uri).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Serving artifact not found",
    ):
        publish_serving_release(
            models_path=str(models_path),
            registration=build_registration(),
            promotion=build_promotion(),
            task_type=TASK_TYPE,
            model_type="xgboost",
            sources=sources,
            metadata=METADATA,
            release_id="release-missing",
        )

    assert not (
        models_path
        / "serving_releases"
        / "release-missing"
    ).exists()