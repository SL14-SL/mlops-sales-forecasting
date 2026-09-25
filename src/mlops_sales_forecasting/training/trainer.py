from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from mlops_sales_forecasting.data.contracts import (
    DatasetSplits,
)
from mlops_sales_forecasting.tracking.mlflow import (
    get_active_training_run_id,
)
from mlops_sales_forecasting.training.contracts import (
    TrainingResult,
)
from mlops_sales_forecasting.training.dataset import (
    normalize_feature_dtypes,
)
from mlops_sales_forecasting.training.model_factory import (
    build_model,
    fit_model,
)
from mlops_sales_forecasting.training.target_transform import (
    inverse_transform_target,
    transform_target,
)
from mlops_sales_forecasting.training.utils import (
    build_drop_columns,
)
from mlops_sales_forecasting.training.weighting import (
    build_recency_weights,
)


class RossmannModelTrainer:
    """Train a configured Rossmann forecasting model."""

    def __init__(
        self,
        *,
        is_drift_run: bool = False,
    ) -> None:
        self.is_drift_run = is_drift_run

    def train(
        self,
        datasets: DatasetSplits,
        config: Mapping[str, Any],
    ) -> TrainingResult:
        data_config = config.get(
            "data",
            {},
        )
        training_config = config.get(
            "training",
            {},
        )
        metrics_config = config.get(
            "metrics",
            {},
        )
        configured_model = config.get("model")

        if not isinstance(configured_model, Mapping):
            raise ValueError("Config must contain a valid 'model' section.")

        target_column = data_config.get("target_column")

        if not isinstance(target_column, str) or not target_column:
            raise ValueError("Config must define data.target_column.")

        if target_column not in datasets.train.columns:
            raise ValueError(f"Training split is missing target column '{target_column}'.")

        if target_column not in datasets.validation.columns:
            raise ValueError(f"Validation split is missing target column '{target_column}'.")

        model_config = deepcopy(dict(configured_model))
        model_type = model_config.get("type")

        if not isinstance(model_type, str):
            raise ValueError("Config must define model.type.")

        target_transformation = training_config.get(
            "target_transformation",
            "none",
        )
        evaluate_on_original_scale = bool(
            metrics_config.get(
                "evaluate_on_original_scale",
                True,
            )
        )
        seed = config.get("random_seed")
        drop_columns = build_drop_columns(config)

        train_frame = datasets.train.copy()
        validation_frame = datasets.validation.copy()

        x_train = normalize_feature_dtypes(
            train_frame.drop(
                columns=drop_columns,
                errors="ignore",
            )
        )
        x_validation = normalize_feature_dtypes(
            validation_frame.drop(
                columns=drop_columns,
                errors="ignore",
            )
        )
        y_train = transform_target(
            train_frame[target_column],
            target_transformation,
        )
        y_validation = transform_target(
            validation_frame[target_column],
            target_transformation,
        )

        sample_weight = self._sample_weight(
            train_frame,
            training_config,
        )

        model = build_model(
            model_config,
            seed=(int(seed) if seed is not None else None),
        )
        fit_model(
            model=model,
            model_type=model_type,
            X_train=x_train,
            y_train=y_train,
            X_val=x_validation,
            y_val=y_validation,
            sample_weight=sample_weight,
        )

        predictions = np.asarray(model.predict(x_validation))

        if evaluate_on_original_scale:
            metric_predictions = inverse_transform_target(
                predictions,
                target_transformation,
            )
            metric_actuals = validation_frame[target_column].to_numpy()
        else:
            metric_predictions = predictions
            metric_actuals = np.asarray(y_validation)

        validation_rmse = float(
            np.sqrt(
                mean_squared_error(
                    metric_actuals,
                    metric_predictions,
                )
            )
        )

        parameters = {
            "model_type": model_type,
            "target_column": target_column,
            "target_transformation": (target_transformation),
            "evaluate_on_original_scale": (evaluate_on_original_scale),
            "training_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "recency_weighting_enabled": (sample_weight is not None),
            **{
                f"model_{name}": value
                for name, value in model_config.get(
                    "params",
                    {},
                ).items()
            },
        }

        return TrainingResult(
            model=model,
            run_id=get_active_training_run_id(),
            metrics={
                "validation_rmse": (validation_rmse),
            },
            parameters=parameters,
            input_example=x_train.head(5).copy(),
        )

    def _sample_weight(
        self,
        train_frame: pd.DataFrame,
        training_config: Mapping[str, Any],
    ) -> np.ndarray | None:
        weighting_config = training_config.get(
            "recency_weighting",
            {},
        )

        if not (
            self.is_drift_run
            and weighting_config.get(
                "enabled_for_drift",
                False,
            )
        ):
            return None

        promo_column = str(
            weighting_config.get(
                "promo_column",
                "Promo",
            )
        )

        if "Date" not in train_frame.columns:
            raise ValueError("Date is required for recency weighting.")

        if promo_column not in train_frame.columns:
            raise ValueError(f"Promo column '{promo_column}' is missing.")

        return build_recency_weights(
            dates=train_frame["Date"],
            promo_values=train_frame[promo_column],
            weighting_config=dict(weighting_config),
        )
