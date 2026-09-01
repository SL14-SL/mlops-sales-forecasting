
from prefect import task, get_run_logger

from src.training.train import train
from src.training.register import register_model


@task(name="Model Training")
def task_train(is_drift_run: bool):
    """
    Train a candidate model and return its MLflow run identifier.

    Args:
        is_drift_run: Whether training was triggered by detected drift.

    Returns:
        The MLflow run ID of the trained candidate.
    """
    p_logger = get_run_logger()

    p_logger.info(
        "Triggering model training task | drift_run=%s",
        is_drift_run,
    )

    model, run_id = train(
        is_drift_run=is_drift_run,
    )

    return run_id

@task(name="Final Model Refit")
def task_final_refit(
    candidate_run_id: str,
    is_drift_run: bool,
) -> dict[str,str]:
    """
    Refit an accepted candidate on train and validation data.
    """
    p_logger = get_run_logger()

    p_logger.info(
        "Starting final refit | "
        f"candidate_run_id={candidate_run_id} | "
        f"drift_run={is_drift_run}"
    )

    _, final_run_id = train(
        is_drift_run=is_drift_run,
        run_role="final_refit",
        candidate_run_id=candidate_run_id,
    )

    model_version = register_model(
        final_run_id, 
        alias="champion",
    )

    p_logger.info(
        "Final refit registered as Champion | "
        f"final_run_id={final_run_id}"
    )

    return {
        "run_id": final_run_id,
        "model_version": str(model_version.version),
    }