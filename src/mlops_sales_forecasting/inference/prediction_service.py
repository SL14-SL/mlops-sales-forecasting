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
from mlops_sales_forecasting.monitoring.inference_store import (
    record_inference_batch,
)
from mlops_sales_forecasting.training.target_transform import (
    inverse_transform_target,
)
from mlops_sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)


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
            raise ValueError("Model output must contain column 'prediction' or exactly one column.")
    elif isinstance(raw_output, pd.Series):
        series = raw_output
    else:
        output_frame = pd.DataFrame(raw_output)

        if len(output_frame.columns) != 1:
            raise ValueError("Model output must be one-dimensional.")

        series = output_frame.iloc[:, 0]

    try:
        values = [float(value) for value in series.tolist()]
    except (TypeError, ValueError) as exc:
        raise ValueError("Model output contains non-numeric values.") from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Model output contains non-finite values.")

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

    native_columns = getattr(
        model,
        "input_columns",
        None,
    )

    if (
        isinstance(native_columns, (list, tuple))
        and native_columns
        and all(isinstance(column, str) and column for column in native_columns)
    ):
        return list(native_columns)

    if metadata is None:
        raise ValueError("Loaded model has no MLflow metadata.")

    input_schema = metadata.get_input_schema()

    if input_schema is None:
        raise ValueError("Loaded model has no input signature.")

    columns = input_schema.input_names()

    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) and column for column in columns)
    ):
        raise ValueError("Loaded model has no valid input columns.")

    return columns


def _align_features_for_model(
    features: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """Select model features in MLflow-signature order."""
    expected_columns = _model_input_columns(model)
    missing_columns = [column for column in expected_columns if column not in features.columns]

    if missing_columns:
        raise ValueError(f"Inference features are missing model columns: {missing_columns}.")

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
        raise ValueError("Prediction input failed schema validation.") from exc


def _configured_drift_features(
    config: Mapping[str, Any],
) -> list[str]:
    """Return the configured inference feature allowlist."""
    monitoring = config.get(
        "monitoring",
        {},
    )

    if not isinstance(monitoring, Mapping):
        return []

    feature_drift = monitoring.get(
        "feature_drift",
        {},
    )

    if not isinstance(feature_drift, Mapping):
        return []

    features: list[str] = []

    for setting_name in (
        "numeric_features",
        "categorical_features",
    ):
        configured_features = feature_drift.get(
            setting_name,
            [],
        )

        if not isinstance(
            configured_features,
            list,
        ):
            raise ValueError(
                f"Monitoring feature list must be a list: monitoring.feature_drift.{setting_name}."
            )

        for feature in configured_features:
            if not isinstance(feature, str) or not feature:
                raise ValueError("Monitoring feature names must be non-empty strings.")

            if feature not in features:
                features.append(feature)

    return features


def _record_inference_batch(
    *,
    validated_input: pd.DataFrame,
    inference_features: pd.DataFrame,
    response: PredictionResponse,
    request_id: str | None,
    config: Mapping[str, Any],
) -> None:
    """Persist inference data according to monitoring policy."""
    monitoring = config.get(
        "monitoring",
        {},
    )

    if not isinstance(monitoring, Mapping):
        return

    logging_config = monitoring.get(
        "inference_logging",
        {},
    )

    if not isinstance(logging_config, Mapping):
        return

    if not bool(
        logging_config.get(
            "enabled",
            False,
        )
    ):
        return

    fail_on_error = bool(
        logging_config.get(
            "fail_on_error",
            False,
        )
    )

    try:
        paths = config.get(
            "paths",
            {},
        )

        if not isinstance(paths, Mapping):
            raise ValueError("Config must contain a valid 'paths' section.")

        predictions_path = paths.get("predictions")

        if not isinstance(predictions_path, str) or not predictions_path:
            raise ValueError("Config must contain a non-empty 'paths.predictions' value.")

        resolved_request_id = request_id or "unassigned"

        record_inference_batch(
            validated_input=validated_input,
            inference_features=inference_features,
            predictions=[result.prediction for result in response.predictions],
            release_id=response.release_id,
            request_id=resolved_request_id,
            feature_allowlist=(_configured_drift_features(config)),
            predictions_path=predictions_path,
        )
    except Exception:
        if fail_on_error:
            raise

        logger.exception(
            "Could not persist inference monitoring batch | request_id=%s",
            request_id or "unassigned",
        )


def predict_with_bundle(
    request: PredictionRequest,
    bundle: ServingBundle,
    config: Mapping[str, Any],
    *,
    request_id: str | None = None,
) -> PredictionResponse:
    """Generate Rossmann forecasts with one serving bundle."""
    validated_frame = _validate_prediction_input(request.inputs)
    open_flags = resolve_open_flags(validated_frame)

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

    transformed_predictions = _output_values(bundle.model.predict(model_input))

    if len(transformed_predictions) != len(request.inputs):
        raise ValueError("Model output length does not match the prediction input count.")

    predictions: list[PredictionResult] = []

    for row_index, transformed_prediction in enumerate(transformed_predictions):
        prediction = float(
            inverse_transform_target(
                transformed_prediction,
                bundle.target_transformation,
            )
        )

        if open_flags is not None:
            prediction = apply_forecasting_business_rules(
                prediction,
                open_flags[row_index],
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

    response = PredictionResponse(
        release_id=bundle.release_id,
        predictions=predictions,
    )

    _record_inference_batch(
        validated_input=validated_frame,
        inference_features=features,
        response=response,
        request_id=request_id,
        config=config,
    )

    return response


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
        *,
        request_id: str | None = None,
    ) -> PredictionResponse:
        """Run forecasts using the active serving bundle."""
        bundle = self._model_manager.get_bundle()

        return predict_with_bundle(
            request,
            bundle,
            self._config,
            request_id=request_id,
        )
