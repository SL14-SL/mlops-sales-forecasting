from datetime import UTC, datetime

import pytest

from mlops_sales_forecasting.inference.releases.builder import (
    build_serving_release_manifest,
)
from mlops_sales_forecasting.inference.releases.contracts import (
    ArtifactReference,
    TaskType,
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

VALID_CHECKSUM = "a" * 64


def build_registration() -> ModelRegistrationResult:
    return ModelRegistrationResult(
        registered=True,
        run_id="run-7",
        model_name="example-model",
        model_version="7",
        model_uri="models:/example-model/7",
    )


def build_promotion(
    *,
    promote: bool = True,
    assignment: bool = True,
) -> PromotionOutcome:
    registration = build_registration()

    champion_assignment = (
        AliasAssignment(
            model_name=registration.model_name,
            model_version="7",
            alias=ModelAlias.CHAMPION,
        )
        if assignment
        else None
    )

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
        champion_assignment=champion_assignment,
    )


def artifact(path: str) -> ArtifactReference:
    return ArtifactReference(
        path=path,
        sha256=VALID_CHECKSUM,
    )



def build_artifacts() -> dict[
    str,
    ArtifactReference,
]:
    return {
        "store_metadata": artifact(
            "store_metadata.parquet"
        ),
        "store_state": artifact(
            "store_state.json"
        ),
        "known_calendar": artifact(
            "known_calendar.parquet"
        ),
    }


def build_metadata() -> dict:
    return {
        "target_transformation": "log1p",
    }


TASK_TYPE = TaskType.FORECASTING



def test_builds_serving_release_manifest() -> None:
    manifest = build_serving_release_manifest(
        registration=build_registration(),
        promotion=build_promotion(),
        task_type=TASK_TYPE,
        model_type="xgboost",
        artifacts=build_artifacts(),
        metadata=build_metadata(),
        dataset_version="dataset-v3",
        config_hash="config-hash",
        git_commit="abc123",
        release_id="release-7",
        clock=lambda: datetime(
            2026,
            9,
            16,
            8,
            0,
            tzinfo=UTC,
        ),
    )

    assert manifest.release_id == "release-7"
    assert (
        manifest.created_at_utc
        == "2026-09-16T08:00:00+00:00"
    )
    assert manifest.task_type is TASK_TYPE
    assert manifest.model.name == "example-model"
    assert manifest.model.version == "7"
    assert manifest.model.run_id == "run-7"
    assert (
        manifest.model.uri
        == "models:/example-model/7"
    )
    assert manifest.model.model_type == "xgboost"
    assert manifest.artifacts == build_artifacts()
    assert manifest.metadata == build_metadata()
    assert manifest.dataset_version == "dataset-v3"
    assert manifest.config_hash == "config-hash"
    assert manifest.git_commit == "abc123"


def test_generates_release_id() -> None:
    manifest = build_serving_release_manifest(
        registration=build_registration(),
        promotion=build_promotion(),
        task_type=TASK_TYPE,
        model_type="xgboost",
        artifacts=build_artifacts(),
        metadata=build_metadata(),
        release_id_factory=(
            lambda: "generated-release"
        ),
    )

    assert (
        manifest.release_id
        == "generated-release"
    )


def test_rejects_candidate_without_promotion() -> None:
    with pytest.raises(
        ValueError,
        match="successful champion promotion",
    ):
        build_serving_release_manifest(
            registration=build_registration(),
            promotion=build_promotion(
                promote=False
            ),
            task_type=TASK_TYPE,
            model_type="xgboost",
            artifacts=build_artifacts(),
            metadata=build_metadata(),
        )


def test_rejects_missing_champion_assignment() -> None:
    with pytest.raises(
        ValueError,
        match="no champion alias assignment",
    ):
        build_serving_release_manifest(
            registration=build_registration(),
            promotion=build_promotion(
                assignment=False
            ),
            task_type=TASK_TYPE,
            model_type="xgboost",
            artifacts=build_artifacts(),
            metadata=build_metadata(),
        )


def test_rejects_mismatched_champion_assignment() -> None:
    promotion = build_promotion()
    mismatched_assignment = AliasAssignment(
        model_name="different-model",
        model_version="7",
        alias=ModelAlias.CHAMPION,
    )
    mismatched_promotion = PromotionOutcome(
        decision=promotion.decision,
        previous_champion_version=(
            promotion.previous_champion_version
        ),
        champion_assignment=(
            mismatched_assignment
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        build_serving_release_manifest(
            registration=build_registration(),
            promotion=mismatched_promotion,
            task_type=TASK_TYPE,
            model_type="xgboost",
            artifacts=build_artifacts(),
            metadata=build_metadata(),
        )


def test_rejects_naive_creation_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="must include a timezone",
    ):
        build_serving_release_manifest(
            registration=build_registration(),
            promotion=build_promotion(),
            task_type=TASK_TYPE,
            model_type="xgboost",
            artifacts=build_artifacts(),
            metadata=build_metadata(),
            clock=lambda: datetime(
                2026,
                9,
                16,
                8,
                0,
            ),
        )