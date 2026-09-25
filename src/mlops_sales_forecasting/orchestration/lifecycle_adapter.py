from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from prefect import flow

from ..inference.releases.input_provider import (
    build_serving_release_input,
)
from ..inference.releases.publisher import (
    PublishedServingRelease,
    publish_serving_release,
)
from ..notifications.contracts import (
    NotificationSink,
)
from ..notifications.factory import (
    build_notification_sink,
)
from ..pipeline.runner import (
    TrackedPipelineResult,
)
from ..pipeline.service import TrainingPipeline
from ..tracking.aliases import restore_champion
from ..tracking.lifecycle import (
    CandidateLifecycleResult,
    finalize_configured_model_candidate,
)
from ..tracking.mlflow import (
    log_evaluation_result,
    log_training_result,
    start_training_run,
)
from ..tracking.model_artifact import (
    LoggedModelArtifact,
    log_model_artifact,
)
from .lifecycle_notifications import (
    notify_candidate_outcome,
    notify_lifecycle_failure,
)
from .prefect_adapter import (
    run_prefect_training_pipeline,
)


@dataclass(frozen=True)
class PrefectTrainingLifecycleResult:
    """Complete result of training and model finalization."""

    pipeline: TrackedPipelineResult
    model_artifact: LoggedModelArtifact
    candidate: CandidateLifecycleResult
    serving_release: (
        PublishedServingRelease | None
    )


def _models_path(
    config: Mapping[str, Any],
) -> str:
    """Return the configured serving-model storage path."""

    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError(
            "Config must contain a valid "
            "'paths' section."
        )

    models_path = paths.get("models")

    if (
        not isinstance(models_path, str)
        or not models_path.strip()
        or models_path.startswith("${")
    ):
        raise ValueError(
            "Config path 'models' must be a "
            "resolved non-empty string."
        )

    return models_path


def _publish_promoted_release(
    *,
    pipeline: TrainingPipeline,
    tracked_result: TrackedPipelineResult,
    candidate_result: CandidateLifecycleResult,
) -> PublishedServingRelease | None:
    """Publish a release only for a promoted champion."""

    promotion = candidate_result.promotion

    if (
        promotion is None
        or not promotion.decision.promote
    ):
        return None

    pipeline_result = tracked_result.pipeline

    try:
        release_input = (
            build_serving_release_input(
                provider=(
                    pipeline.release_input_provider
                ),
                training_result=(
                    pipeline_result.training
                ),
                evaluation_result=(
                    pipeline_result.evaluation
                ),
                config=pipeline.config,
            )
        )

        return publish_serving_release(
            models_path=_models_path(
                pipeline.config
            ),
            registration=(
                candidate_result.registration
            ),
            promotion=promotion,
            task_type=release_input.task_type,
            model_type=(
                release_input.model_type
            ),
            sources=release_input.sources,
            metadata=release_input.metadata,
            dataset_version=(
                release_input.dataset_version
            ),
            config_hash=(
                release_input.config_hash
            ),
            git_commit=(
                release_input.git_commit
            ),
        )
    except Exception as release_error:
        try:
            restore_champion(
                registration=(
                    candidate_result.registration
                ),
                previous_champion_version=(
                    promotion
                    .previous_champion_version
                ),
                config=pipeline.config,
            )
        except Exception as restoration_error:
            raise ExceptionGroup(
                "Serving release publication and "
                "champion restoration both failed.",
                [
                    release_error,
                    restoration_error,
                ],
            ) from None

        raise


@flow(
    name="training-model-lifecycle",
    validate_parameters=False,
    persist_result=False,
)
def run_prefect_model_lifecycle(
    *,
    pipeline: TrainingPipeline,
    pipeline_run_id: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_tags: Mapping[str, str] | None = None,
    artifact_path: str = "model",
    notification_sink: (
        NotificationSink | None
    ) = None,
) -> PrefectTrainingLifecycleResult:
    """Run training, finalization and lifecycle notifications."""

    sink = (
        notification_sink
        or build_notification_sink(
            pipeline.config
        )
    )
    active_run_id = (
        pipeline_run_id
        or mlflow_run_name
        or "unassigned"
    )

    try:
        with start_training_run(
            pipeline.config,
            run_name=mlflow_run_name,
            tags=mlflow_tags,
        ) as mlflow_run_id:
            active_run_id = (
                pipeline_run_id
                or mlflow_run_id
            )

            tracked_result = (
                run_prefect_training_pipeline(
                    pipeline=pipeline,
                    run_id=active_run_id,
                )
            )

            pipeline_result = (
                tracked_result.pipeline
            )

            log_training_result(
                pipeline_result.training
            )
            log_evaluation_result(
                pipeline_result.evaluation
            )
            model_artifact = (
                log_model_artifact(
                    logger=(
                        pipeline.model_logger
                    ),
                    training_result=(
                        pipeline_result.training
                    ),
                    config=pipeline.config,
                    artifact_path=artifact_path,
                )
            )

        candidate_result = (
            finalize_configured_model_candidate(
                training_result=(
                    pipeline_result.training
                ),
                evaluation_result=(
                    pipeline_result.evaluation
                ),
                config=pipeline.config,
                artifact_path=artifact_path,
                logged_model_uri=(
                    model_artifact.model_uri
                ),
            )
        )

        serving_release = (
            _publish_promoted_release(
                pipeline=pipeline,
                tracked_result=tracked_result,
                candidate_result=(
                    candidate_result
                ),
            )
        )

        notify_candidate_outcome(
            sink=sink,
            config=pipeline.config,
            candidate=candidate_result,
            serving_release=serving_release,
        )

        return PrefectTrainingLifecycleResult(
            pipeline=tracked_result,
            model_artifact=model_artifact,
            candidate=candidate_result,
            serving_release=serving_release,
        )

    except Exception as lifecycle_error:
        try:
            notify_lifecycle_failure(
                sink=sink,
                config=pipeline.config,
                run_id=active_run_id,
                error=lifecycle_error,
            )
        except Exception as notification_error:
            raise ExceptionGroup(
                "Model lifecycle and failure "
                "notification both failed.",
                [
                    lifecycle_error,
                    notification_error,
                ],
            ) from None

        raise