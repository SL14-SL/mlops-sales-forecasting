import pytest
from pydantic import ValidationError

from mlops_sales_forecasting.api.schema import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)


def test_prediction_request_accepts_horizon() -> None:
    request = PredictionRequest(
        inputs=[
            {
                "entity_id": 1,
                "date": "2026-09-16",
            }
        ],
        horizon=7,
    )

    assert request.horizon == 7
    assert len(request.inputs) == 1


def test_prediction_request_defaults_to_one_step() -> None:
    request = PredictionRequest(
        inputs=[{"entity_id": 1}]
    )

    assert request.horizon == 1


def test_prediction_request_rejects_empty_batch() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(
            inputs=[],
            horizon=1,
        )


def test_prediction_request_rejects_invalid_horizon() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(
            inputs=[{"entity_id": 1}],
            horizon=0,
        )


def test_prediction_response_serializes() -> None:
    response = PredictionResponse(
        release_id="release-1",
        predictions=[
            PredictionResult(
                row_index=0,
                horizon_step=1,
                prediction=123.45,
            )
        ],
    )

    assert response.model_dump() == {
        "status": "success",
        "release_id": "release-1",
        "predictions": [
            {
                "row_index": 0,
                "horizon_step": 1,
                "prediction": 123.45,
            }
        ],
    }
