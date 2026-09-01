from dataclasses import dataclass
import time
from typing import Any

import pandas as pd

from src.data.features.build_features import (
    preprocess_data,
)
from src.inference.adapters import (
    request_to_dataframe,
    resolve_forecasting_store_id,
    resolve_open_flags,
)
from src.inference.forecasting_policy import (
    finalize_forecasting_feature_frame,
    inject_forecasting_state_features,
    merge_request_with_calendar,
    merge_request_with_metadata,
)
from src.inference.pipeline import (
    align_features_for_model,
    apply_prediction_postprocessing,
    validate_prediction_input,
)
from src.inference.serving_bundle import (
    ServingBundle,
)
from src.training.target_transform import (
    inverse_transform_target,
)


@dataclass(frozen=True)
class PredictionExecution:
    """
    Prediction output together with timing, quality and serving-lineage metadata.
    """
    predictions: tuple[float, ...]
    validated_input: pd.DataFrame
    timings_ms: dict[str, float]

    @property
    def unique_stores(self) -> int | None:
        if (
            "Store"
            not in self.validated_input.columns
        ):
            return None

        return int(
            self.validated_input[
                "Store"
            ].nunique()
        )


def _milliseconds_since(
    started_at: float,
) -> float:
    return round(
        (
            time.perf_counter()
            - started_at
        )
        * 1_000,
        2,
    )


def predict_with_bundle(
    *,
    inputs: list[dict[str, Any]],
    bundle: ServingBundle,
) -> PredictionExecution:
    """
    Generate forecasts with one immutable serving bundle.

    The function prepares model input from request rows and release artifacts,
    executes the model, reverses the configured target transformation and returns
    predictions with serving lineage and timing metadata.

    Args:
        bundle: Active model and inference-artifact bundle.
        inputs: Store and date combinations to forecast.

    Returns:
        Prediction results and execution metadata bound to the bundle release.

    Raises:
        ValueError: If inputs or required artifacts are incompatible.
        RuntimeError: If model execution does not produce valid predictions.
    """
    timings: dict[str, float] = {}

    started_at = time.perf_counter()

    input_df = request_to_dataframe(
        inputs
    )

    timings[
        "request_to_dataframe"
    ] = _milliseconds_since(
        started_at
    )

    started_at = time.perf_counter()

    validated_input = (
        validate_prediction_input(
            input_df
        )
    )

    timings[
        "validate_prediction_input"
    ] = _milliseconds_since(
        started_at
    )

    predictions: list[float] = []
    prediction_started_at = (
        time.perf_counter()
    )

    for row in inputs:
        row_df = request_to_dataframe(
            [row]
        )

        row_validated_df = (
            validate_prediction_input(
                row_df
            )
        )

        store_id = (
            resolve_forecasting_store_id(
                row_validated_df
            )
        )

        open_flags = resolve_open_flags(
            row_validated_df
        )

        features_df = (
            merge_request_with_metadata(
                validated_df=(
                    row_validated_df
                ),
                store_metadata=(
                    bundle.store_metadata
                ),
                store_id=store_id,
            )
        )

        features_df = (
            merge_request_with_calendar(
                features_df,
                bundle.known_calendar,
            )
        )

        processed_df = preprocess_data(
            features_df,
            mode="inference",
        )

        processed_df = (
            inject_forecasting_state_features(
                processed_df=processed_df,
                store_state=(
                    bundle.store_state
                ),
                store_id=store_id,
            )
        )

        processed_df = (
            finalize_forecasting_feature_frame(
                processed_df
            )
        )

        processed_df = (
            align_features_for_model(
                processed_df=processed_df,
                model=bundle.model,
                model_type=bundle.model_type,
            )
        )

        raw_predictions = (
            bundle.model.predict(
                processed_df
            )
        )

        row_predictions = [
            float(
                inverse_transform_target(
                    float(prediction),
                    bundle.target_transformation,
                )
            )
            for prediction in raw_predictions
        ]

        row_predictions = (
            apply_prediction_postprocessing(
                row_predictions,
                open_flags,
            )
        )

        predictions.extend(
            row_predictions
        )

    timings[
        "predict_rows_single_logic"
    ] = _milliseconds_since(
        prediction_started_at
    )

    postprocess_started_at = (
        time.perf_counter()
    )

    rounded_predictions = tuple(
        round(
            float(prediction),
            2,
        )
        for prediction in predictions
    )

    timings[
        "postprocess_predictions"
    ] = _milliseconds_since(
        postprocess_started_at
    )

    if len(rounded_predictions) != len(
        inputs
    ):
        raise RuntimeError(
            "Prediction count mismatch: "
            f"got {len(rounded_predictions)} "
            "predictions for "
            f"{len(inputs)} input rows."
        )

    return PredictionExecution(
        predictions=rounded_predictions,
        validated_input=validated_input,
        timings_ms=timings,
    )