from collections.abc import Mapping
from typing import Any

import mlflow.xgboost

from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)


class XGBoostModelArtifactLogger:
    """Log a trained XGBoost model to the active MLflow run."""

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
            raise ValueError("XGBoostModelArtifactLogger requires model.type='xgboost'.")

        data_config = config.get(
            "data",
            {},
        )
        training_config = config.get(
            "training",
            {},
        )

        mlflow.xgboost.log_model(
            training_result.model,
            name=artifact_path,
            metadata={
                "model_type": "xgboost",
                "target_column": str(data_config.get("target_column", "")),
                "target_transformation": str(
                    training_config.get(
                        "target_transformation",
                        "none",
                    )
                ),
            },
        )

        return f"runs:/{training_result.run_id}/{artifact_path}"
