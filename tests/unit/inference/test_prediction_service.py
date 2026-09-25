from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.api.schema import PredictionRequest
from mlops_sales_forecasting.inference.prediction_service import (
    PredictionService,
    predict_with_bundle,
)


def build_bundle() -> MagicMock:
    bundle = MagicMock()
    bundle.release_id = "release-1"
    return bundle


def test_forecasting_prediction() -> None:
    bundle = build_bundle()
    bundle.model.predict.return_value = [
        100.0,
        110.0,
        120.0,
        200.0,
        210.0,
        220.0,
    ]
    request = PredictionRequest(
        inputs=[
            {"entity_id": 1},
            {"entity_id": 2},
        ],
        horizon=3,
    )

    response = predict_with_bundle(
        request,
        bundle,
    )

    assert len(response.predictions) == 6
    assert [
        prediction.prediction
        for prediction in response.predictions
    ] == [
        100.0,
        110.0,
        120.0,
        200.0,
        210.0,
        220.0,
    ]
    assert response.predictions[2].horizon_step == 3
    assert response.predictions[3].row_index == 1
    bundle.model.predict.assert_called_once()

    call = bundle.model.predict.call_args
    assert call.kwargs == {
        "params": {
            "horizon": 3,
        }
    }


def test_forecasting_accepts_named_prediction_column() -> None:
    bundle = build_bundle()
    bundle.model.predict.return_value = pd.DataFrame(
        {
            "prediction": [100.0],
        }
    )
    request = PredictionRequest(
        inputs=[{"entity_id": 1}],
        horizon=1,
    )

    response = predict_with_bundle(request, bundle)

    assert response.predictions[0].prediction == 100.0


def test_forecasting_rejects_wrong_output_length() -> None:
    bundle = build_bundle()
    bundle.model.predict.return_value = [100.0]
    request = PredictionRequest(
        inputs=[{"entity_id": 1}],
        horizon=2,
    )

    with pytest.raises(
        ValueError,
        match="output length does not match",
    ):
        predict_with_bundle(request, bundle)


def test_prediction_service_uses_active_bundle() -> None:
    bundle = build_bundle()

    bundle.model.predict.return_value = [100.0]
    request = PredictionRequest(
        inputs=[{"entity_id": 1}],
        horizon=1,
    )

    model_manager = MagicMock()
    model_manager.get_bundle.return_value = bundle
    service = PredictionService(model_manager)

    response = service.predict(request)

    assert response.release_id == "release-1"
    model_manager.get_bundle.assert_called_once_with()