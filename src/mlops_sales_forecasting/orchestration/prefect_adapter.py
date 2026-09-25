from prefect import flow, get_run_logger

from ..pipeline.runner import (
    TrackedPipelineResult,
)
from ..pipeline.service import TrainingPipeline
from ..pipeline.status import PipelineRunStatus


@flow(
    name="training-pipeline",
    validate_parameters=False,
    persist_result=False,
)
def run_prefect_training_pipeline(
    *,
    pipeline: TrainingPipeline,
    run_id: str | None = None,
) -> TrackedPipelineResult:
    """Execute the configured training pipeline with Prefect."""
    logger = get_run_logger()

    logger.info(
        "Starting training pipeline | run_id=%s",
        run_id or "generated",
    )

    try:
        result = pipeline.run(
            run_id=run_id
        )
    except Exception:
        logger.exception(
            "Training pipeline failed | run_id=%s",
            run_id or "generated",
        )
        raise

    final_run = result.run

    if (
        final_run.status
        is PipelineRunStatus.REJECTED
    ):
        logger.warning(
            "Training candidate rejected | "
            "run_id=%s | reasons=%s",
            final_run.run_id,
            list(final_run.rejection_reasons),
        )
    else:
        logger.info(
            "Training pipeline completed | "
            "run_id=%s | status=%s | "
            "duration_seconds=%s",
            final_run.run_id,
            final_run.status.value,
            final_run.duration_seconds,
        )

    return result