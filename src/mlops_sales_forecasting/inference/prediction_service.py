import math

import pandas as pd

from ..api.schema import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from .model_manager import ModelManager
from .serving_bundle import ServingBundle


def _output_values(
    raw_output,
    *,
    preferred_column: str,
) -> list[float]:
    """Normalize model output into finite floating-point values."""
    if isinstance(raw_output, pd.DataFrame):
        if preferred_column in raw_output.columns:
            series = raw_output[preferred_column]
        elif len(raw_output.columns) == 1:
            series = raw_output.iloc[:, 0]
        else:
            raise ValueError(
                "Model output must contain column "
                f"'{preferred_column}' or exactly one column."
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
        values = [float(value) for value in series.tolist()]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Model output contains non-numeric values."
        ) from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "Model output contains non-finite values."
        )

    return values



def predict_with_bundle(
    request: PredictionRequest,
    bundle: ServingBundle,
) -> PredictionResponse:
    """Run forecasting for one validated serving bundle."""
    frame = pd.DataFrame(request.inputs)
    raw_output = bundle.model.predict(
        frame,
        params={
            "horizon": request.horizon,
        },
    )
    forecast_values = _output_values(
        raw_output,
        preferred_column="prediction",
    )
    expected_values = len(request.inputs) * request.horizon

    if len(forecast_values) != expected_values:
        raise ValueError(
            "Forecast model output length does not match "
            "input count multiplied by horizon."
        )

    predictions = []

    for row_index in range(len(request.inputs)):
        for horizon_step in range(1, request.horizon + 1):
            output_index = (
                row_index * request.horizon
                + horizon_step
                - 1
            )
            predictions.append(
                PredictionResult(
                    row_index=row_index,
                    horizon_step=horizon_step,
                    prediction=forecast_values[output_index],
                )
            )

    return PredictionResponse(
        release_id=bundle.release_id,
        predictions=predictions,
    )


class PredictionService:
    """Run predictions against the currently active bundle."""

    def __init__(
        self,
        model_manager: ModelManager,
    ) -> None:
        self._model_manager = model_manager

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """Run a prediction using the active serving bundle."""
        bundle = self._model_manager.get_bundle()
        return predict_with_bundle(
            request,
            bundle,
        )