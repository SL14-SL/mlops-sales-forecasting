from __future__ import annotations

from prefect import flow, get_run_logger

from flows.training_flow import training_pipeline
from src.monitoring.trigger import should_retrain


@flow(name="Auto Retrain Decision Flow")
def auto_retrain_flow() -> str:
    logger = get_run_logger()

    if not should_retrain():
        logger.info("No new data and no drift detected. Skipping retraining.")
        return "skipped"

    logger.info("Retraining conditions met. Triggering training pipeline.")
    training_pipeline(force_run=True)
    return "retrained"


if __name__ == "__main__":
    auto_retrain_flow()