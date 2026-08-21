from prefect import task, get_run_logger

from src.training.train import train
from src.training.evaluate import compare_models, evaluate_model, champion_exists
from src.training.register import register_model


@task(name="Evaluate Current Champion")
def task_evaluate_champion():
    p_logger = get_run_logger()
    p_logger.info("Evaluating current champion for dashboard continuity.")
    try:
        rmse = evaluate_model(model_alias="champion")
        print(f"Champion RMSE: {rmse}")
        return rmse
    except Exception as e:
        p_logger.warning(f"Could not evaluate champion: {e}")
        return None


@task(name="Candidate Evaluation")
def task_eval_and_reg(
    new_run_id: str,
) -> bool:
    """
    Evaluate the Candidate against the current Champion.

    The Candidate is accepted only when it passes every configured promotion
    gate. Rejected Candidates are retained in MLflow under the Challenger
    alias for audit, analysis and possible shadow evaluation.

    Comparison or policy errors propagate and block all registry changes.
    """
    p_logger = get_run_logger()

    candidate_accepted, metrics = (
        compare_models(
            new_run_id
        )
    )

    candidate_metrics = metrics.get(
        "candidate_metrics",
        {},
    )

    champion_metrics = metrics.get(
        "champion_metrics",
        {},
    )

    promotion_decision = metrics.get(
        "promotion_decision",
        {},
    )

    p_logger.info(
        "Promotion policy evaluated | "
        "candidate_run_id=%s | "
        "accepted=%s | "
        "candidate_rmse=%s | "
        "champion_rmse=%s | "
        "reasons=%s",
        new_run_id,
        promotion_decision.get(
            "accepted"
        ),
        candidate_metrics.get(
            "overall_rmse"
        ),
        champion_metrics.get(
            "overall_rmse"
        ),
        promotion_decision.get(
            "reasons",
            [],
        ),
    )

    # Compatibility output for the current lifecycle scripts.
    if "rmse_euro" in metrics:
        print(
            "Challenger RMSE: "
            f"{metrics['rmse_euro']}"
        )

    if candidate_accepted:
        p_logger.info(
            "Candidate passed all promotion gates | "
            f"candidate_run_id={new_run_id}"
        )
        return True

    p_logger.info(
        "Candidate rejected by promotion policy. "
        "Registering it as Challenger | "
        f"candidate_run_id={new_run_id} | "
        f"reasons={promotion_decision.get('reasons', [])}"
    )

    register_model(
        new_run_id,
        alias="challenger",
    )

    return False

@task(name="Bootstrap Initial Champion")
def task_bootstrap_champion(
    candidate_run_id: str,
    is_drift_run: bool,
) -> dict[str,str]:
    """
    Create the first Champion in an empty model registry.

    Bootstrap is rejected when a Champion already exists.
    """
    p_logger = get_run_logger()

    if champion_exists():
        raise RuntimeError(
            "Bootstrap rejected: a Champion already exists."
        )

    p_logger.info(
        "No Champion exists. Starting explicit initial bootstrap | "
        f"candidate_run_id={candidate_run_id}"
    )

    _, final_run_id = train(
        is_drift_run=is_drift_run,
        run_role="final_refit",
        candidate_run_id=candidate_run_id,
    )

    # Check again immediately before changing the alias.
    # This reduces the risk of two concurrent bootstrap runs.
    if champion_exists():
        raise RuntimeError(
            "Bootstrap aborted: a Champion was created concurrently."
        )

    model_version = register_model(
        final_run_id, 
        alias="champion",
    )

    p_logger.info(
        "Initial Champion created | "
        f"candidate_run_id={candidate_run_id} | "
        f"final_run_id={final_run_id}"
    )

    return {
        "run_id": final_run_id,
        "model_version": str(model_version.version),
    }
