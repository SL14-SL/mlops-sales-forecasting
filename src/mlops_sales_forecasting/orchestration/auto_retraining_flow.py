from typing import Any

from prefect import flow, get_run_logger

from ..configs.loader import load_config
from .retraining_service import (
    run_auto_retraining,
)


@flow(
    name="mlops-sales-forecasting-auto-retraining",
    validate_parameters=False,
    persist_result=False,
)
def auto_retraining_flow() -> dict[str, Any]:
    """Evaluate signals and conditionally retrain the model."""
    logger = get_run_logger()
    config = load_config()

    result = run_auto_retraining(config=config)

    logger.info(
        "Auto-retraining cycle completed | "
        "status=%s | decision_id=%s | "
        "candidate_run_id=%s | "
        "champion_promoted=%s | reasons=%s",
        result.status,
        result.decision_id,
        result.candidate_run_id,
        result.champion_promoted,
        list(result.reasons),
    )

    return result.to_dict()


if __name__ == "__main__":
    auto_retraining_flow()
