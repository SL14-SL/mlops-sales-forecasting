from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error

from mlops_sales_forecasting.data.contracts import (
    DatasetSplits,
)
from mlops_sales_forecasting.training.contracts import (
    EvaluationResult,
    TrainingResult,
)
from mlops_sales_forecasting.training.dataset import (
    normalize_feature_dtypes,
)
from mlops_sales_forecasting.training.evaluate_metrics import (
    align_features_for_evaluation,
    calculate_promotion_metrics,
)
from mlops_sales_forecasting.training.target_transform import (
    inverse_transform_target,
)
from mlops_sales_forecasting.training.utils import (
    build_drop_columns,
)


class RossmannModelEvaluator:
    """Evaluate a Rossmann candidate on chronological validation data."""

    def evaluate(
        self,
        training_result: TrainingResult,
        datasets: DatasetSplits,
        config: Mapping[str, Any],
    ) -> EvaluationResult:
        data_config = config.get(
            "data",
            {},
        )
        training_config = config.get(
            "training",
            {},
        )
        promotion_config = config.get(
            "promotion",
            {},
        )

        target_column = data_config.get("target_column")

        if not isinstance(target_column, str) or not target_column:
            raise ValueError("Config must define data.target_column.")

        validation = datasets.validation.copy()

        if target_column not in validation.columns:
            raise ValueError(f"Validation split is missing target column '{target_column}'.")

        features = validation.drop(
            columns=build_drop_columns(config),
            errors="ignore",
        )
        features = normalize_feature_dtypes(features)
        features = align_features_for_evaluation(
            training_result.model,
            features,
        )

        raw_predictions = np.asarray(training_result.model.predict(features))
        target_transformation = training_config.get(
            "target_transformation",
            "none",
        )
        predictions = np.asarray(
            inverse_transform_target(
                raw_predictions,
                target_transformation,
            ),
            dtype=float,
        )
        actual = validation[target_column].astype(float)

        promotion_metrics, segment_rows = calculate_promotion_metrics(
            y_true=actual,
            predictions=predictions,
            evaluation_frame=validation,
        )

        metrics = {
            "rmse": promotion_metrics["overall_rmse"],
            "mae": float(
                mean_absolute_error(
                    actual,
                    predictions,
                )
            ),
            **promotion_metrics,
        }

        reasons: list[str] = []

        minimum_validation_rows = int(
            promotion_config.get(
                "minimum_validation_rows",
                1,
            )
        )

        if len(validation) < minimum_validation_rows:
            reasons.append("Validation dataset contains fewer rows than required.")

        minimum_segment_rows = int(
            promotion_config.get(
                "minimum_segment_rows",
                1,
            )
        )

        for segment in promotion_config.get(
            "required_segments",
            [
                "promo",
                "non_promo",
            ],
        ):
            row_count = segment_rows.get(
                str(segment),
                0,
            )

            if row_count < minimum_segment_rows:
                reasons.append(f"Segment '{segment}' contains fewer rows than required.")

        return EvaluationResult(
            metrics=metrics,
            approved=not reasons,
            reasons=tuple(reasons),
        )
