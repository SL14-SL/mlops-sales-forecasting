import mlflow

from copy import deepcopy

from src.configs.loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")



def build_final_refit_model_config(
    candidate_run_id: str,
) -> tuple[dict, int | None]:
    """
    Build a final model configuration using the candidate's best iteration.

    Early stopping is removed because the final model is trained on all
    available observations without a separate validation set.
    """
    model_cfg = deepcopy(TRAIN_CFG["model"])
    params = model_cfg.setdefault("params", {})

    if model_cfg["type"] != "xgboost":
        return model_cfg, None

    candidate_uri = f"runs:/{candidate_run_id}/model"
    candidate_model = mlflow.xgboost.load_model(candidate_uri)

    best_iteration = None

    try:
        best_iteration = int(candidate_model.best_iteration)
    except (AttributeError, TypeError, ValueError):
        logger.warning(
            "Candidate model does not expose a best iteration. "
            "Using configured n_estimators for final refit."
        )

    params.pop("early_stopping_rounds", None)

    if best_iteration is not None:
        params["n_estimators"] = best_iteration + 1

    return model_cfg, best_iteration