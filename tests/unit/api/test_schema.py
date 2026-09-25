import pytest
from pydantic import ValidationError

from mlops_sales_forecasting.api.schema import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)


def test_prediction_request_accepts_rossmann_rows() -> None:
    request = PredictionRequest(
        inputs=[
            {
                "Store": 1,
                "Date": "2026-09-25",
                "Open": 1,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            },
        ]
    )

    assert len(request.inputs) == 1
    assert request.inputs[0]["Store"] == 1


def test_prediction_request_rejects_empty_batch() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(
            inputs=[],
        )


def test_prediction_request_rejects_horizon() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(
            inputs=[
                {
                    "Store": 1,
                },
            ],
            horizon=7,
        )


def test_prediction_response_serializes() -> None:
    response = PredictionResponse(
        release_id="release-1",
        predictions=[
            PredictionResult(
                row_index=0,
                horizon_step=1,
                prediction=123.45,
            ),
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
            },
        ],
    }