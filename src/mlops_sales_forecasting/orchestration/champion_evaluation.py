from collections.abc import (
    Callable,
    Mapping,
)
from typing import Any

from ..data.contracts import DatasetSplits
from ..inference.model_loader import (
    LoadedXGBoostModel,
    load_xgboost_model,
)
from ..tracking.promotion import ChampionMetricsProvider
from ..training.contracts import (
    ModelEvaluator,
    TrainingResult,
)

ChampionModelLoader = Callable[
    [str],
    LoadedXGBoostModel,
]


def build_champion_metrics_provider(
    *,
    evaluator: ModelEvaluator,
    datasets: DatasetSplits,
    config: Mapping[str, Any],
    model_loader: ChampionModelLoader = (load_xgboost_model),
) -> ChampionMetricsProvider:
    """
    Build a provider that evaluates a champion on the current holdout.

    The returned callable accepts an immutable MLflow model-version URI
    and returns metrics calculated with the same evaluator and validation
    split as the candidate.
    """

    def evaluate_champion(
        model_uri: str,
    ) -> Mapping[str, float]:
        if not isinstance(model_uri, str) or not model_uri.strip():
            raise ValueError("Champion model URI must be a non-empty string.")

        loaded_model = model_loader(model_uri)
        champion_result = TrainingResult(
            model=loaded_model.estimator,
            run_id=("paired-champion-evaluation"),
            metrics={},
        )

        evaluation = evaluator.evaluate(
            champion_result,
            datasets,
            config,
        )

        if not evaluation.approved:
            raise ValueError(
                "Champion could not be evaluated "
                "on the current validation split: " + "; ".join(evaluation.reasons)
            )

        return dict(evaluation.metrics)

    return evaluate_champion
