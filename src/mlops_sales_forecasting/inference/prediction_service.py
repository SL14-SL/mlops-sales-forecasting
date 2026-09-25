import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pandera.pandas as pa

from mlops_sales_forecasting.api.schema import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from mlops_sales_forecasting.data.validation.validate import (
    validate_inference,
)
from mlops_sales_forecasting.inference.adapters import (
    request_to_dataframe,
    resolve_open_flags,
)
from mlops_sales_forecasting.inference.forecasting_policy import (
    apply_forecasting_business_rules,
)
from mlops_sales_forecasting.inference.forecasting_provider import (
    build_forecasting_inference_features,
)
from mlops_sales_forecasting.inference.model_manager import (
    ModelManager,
)
from mlops_sales_forecasting.inference.serving_bundle import (
    ServingBundle,
)
from mlops_sales_forecasting.training.target_transform import (
    inverse_transform_target,
)


def _output_values(
    raw_output: Any,
) -> list[float]:
    """Normalize model output into finite float values."""
    if isinstance(raw_output, pd.DataFrame):
        if "prediction" in raw_output.columns:
            series = raw_output["prediction"]
        elif len(raw_output.columns) == 1:
            series = raw_output.iloc[:, 0]
        else:
            raise ValueError(
                "Model output must contain column "
                "'prediction' or exactly one column."
            )
    elif isinstance(raw_output, pd.Series):
        series = raw_output
    else:
        output_frame = pd.DataFrame(raw_output)

        if len(output_frame.columns) != 1:
            raise ValueError(
                "Model output must be one-dimensional."
            )

        series = output_frame.iloc[:, 0]

    try:
        values = [
            float(value)
            for value in series.tolist()
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Model output contains non-numeric values."
        ) from exc

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Model output contains non-finite values."
        )

    return values


def _model_input_columns(
    model: Any,
) -> list[str]:
    """Return ordered input columns from the MLflow signature."""
    metadata = getattr(
        model,
        "metadata",
        None,
    )

    if metadata is None:
        raise ValueError(
            "Loaded model has no MLflow metadata."
        )

    input_schema = metadata.get_input_schema()

    if input_schema is None:
        raise ValueError(
            "Loaded model has no input signature."
        )

    columns = input_schema.input_names()

    if (
        not isinstance(columns, list)
        or not columns
        or not all(
            isinstance(column, str) and column
            for column in columns
        )
    ):
        raise ValueError(
            "Loaded model has no valid input columns."
        )

    return columns


def _align_features_for_model(
    features: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """Select model features in MLflow-signature order."""
    expected_columns = _model_input_columns(
        model
    )
    missing_columns = [
        column
        for column in expected_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            "Inference features are missing model columns: "
            f"{missing_columns}."
        )

    return features.loc[
        :,
        expected_columns,
    ]


def _validate_prediction_input(
    inputs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert and validate API input records."""
    input_frame = request_to_dataframe(inputs)

    try:
        return validate_inference(input_frame)
    except pa.errors.SchemaError as exc:
        raise ValueError(
            "Prediction input failed schema validation."
        ) from exc


def predict_with_bundle(
    request: PredictionRequest,
    bundle: ServingBundle,
    config: Mapping[str, Any],
) -> PredictionResponse:
    """Generate Rossmann forecasts with one serving bundle."""
    validated_frame = _validate_prediction_input(
        request.inputs
    )
    open_flags = resolve_open_flags(
        validated_frame
    )

    features = build_forecasting_inference_features(
        validated_frame=validated_frame,
        store_metadata=bundle.store_metadata,
        store_state=bundle.store_state,
        known_calendar=bundle.known_calendar,
        config=config,
    )
    model_input = _align_features_for_model(
        features,
        bundle.model,
    )

    transformed_predictions = _output_values(
        bundle.model.predict(model_input)
    )

    if len(transformed_predictions) != len(
        request.inputs
    ):
        raise ValueError(
            "Model output length does not match "
            "the prediction input count."
        )

    predictions: list[PredictionResult] = []

    for row_index, transformed_prediction in enumerate(
        transformed_predictions
    ):
        prediction = float(
            inverse_transform_target(
                transformed_prediction,
                bundle.target_transformation,
            )
        )

        if open_flags is not None:
            prediction = (
                apply_forecasting_business_rules(
                    prediction,
                    open_flags[row_index],
                )
            )

        predictions.append(
            PredictionResult(
                row_index=row_index,
                horizon_step=1,
                prediction=round(
                    prediction,
                    2,
                ),
            )
        )

    return PredictionResponse(
        release_id=bundle.release_id,
        predictions=predictions,
    )


class PredictionService:
    """Run Rossmann forecasts against the active bundle."""

    def __init__(
        self,
        model_manager: ModelManager,
        config: Mapping[str, Any],
    ) -> None:
        self._model_manager = model_manager
        self._config = config

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """Run forecasts using the active serving bundle."""
        bundle = self._model_manager.get_bundle()

        return predict_with_bundle(
            request,
            bundle,
            self._config,
        )