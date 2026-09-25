from collections.abc import Mapping
from typing import Any

import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature

from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)


class XGBoostModelArtifactLogger:
    """Log a trained XGBoost model with its input signature."""

    def log_model(
        self,
        training_result: TrainingResult,
        *,
        artifact_path: str,
        config: Mapping[str, Any],
    ) -> str:
        model_config = config.get(
            "model",
            {},
        )
        model_type = model_config.get("type")

        if model_type != "xgboost":
            raise ValueError(
                "XGBoostModelArtifactLogger requires "
                "model.type='xgboost'."
            )

        input_example = training_result.input_example

        if (
            not isinstance(input_example, pd.DataFrame)
            or input_example.empty
        ):
            raise ValueError(
                "XGBoost model logging requires a "
                "non-empty pandas input example."
            )

        predictions = training_result.model.predict(
            input_example
        )
        signature = infer_signature(
            input_example,
            predictions,
        )

        data_config = config.get(
            "data",
            {},
        )
        training_config = config.get(
            "training",
            {},
        )
        model_info = mlflow.xgboost.log_model(
            training_result.model,
            name=artifact_path,
            signature=signature,
            metadata={
                "model_type": "xgboost",
                "target_column": str(
                    data_config.get(
                        "target_column",
                        "",
                    )
                ),
                "target_transformation": str(
                    training_config.get(
                        "target_transformation",
                        "none",
                    )
                ),
            },
        )

        return model_info.model_uri