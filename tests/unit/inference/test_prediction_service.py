from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mlops_sales_forecasting.api.schema import (
    PredictionRequest,
)
from mlops_sales_forecasting.inference import (
    prediction_service,
)
from mlops_sales_forecasting.inference.prediction_service import (
    PredictionService,
    predict_with_bundle,
)


def config() -> dict:
    return {
        "data": {
            "target_column": "Sales",
            "time_column": "Date",
            "id_columns": [
                "Store",
            ],
        },
        "features": {
            "lag_features": {
                "lags": [
                    1,
                    7,
                ],
                "rolling_windows": [
                    7,
                ],
            },
        },
    }


def request() -> PredictionRequest:
    return PredictionRequest(
        inputs=[
            {
                "Store": 1,
                "Date": "2026-09-25",
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            },
            {
                "Store": 2,
                "Date": "2026-09-25",
                "Open": 0,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            },
        ],
    )


def build_bundle() -> MagicMock:
    bundle = MagicMock()
    bundle.release_id = "release-1"
    bundle.target_transformation = "log1p"
    bundle.store_metadata = pd.DataFrame()
    bundle.store_state = {}
    bundle.known_calendar = pd.DataFrame()

    input_schema = MagicMock()
    input_schema.input_names.return_value = [
        "Store",
        "feature",
    ]
    bundle.model.metadata.get_input_schema.return_value = (
        input_schema
    )

    return bundle


def test_forecasting_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle()
    bundle.model.predict.return_value = np.log1p(
        [
            100.0,
            200.0,
        ]
    )

    features = pd.DataFrame(
        {
            "feature": [
                1.5,
                2.5,
            ],
            "extra": [
                99,
                99,
            ],
            "Store": [
                1,
                2,
            ],
        }
    )
    build_features = MagicMock(
        return_value=features
    )

    monkeypatch.setattr(
        prediction_service,
        "build_forecasting_inference_features",
        build_features,
    )

    response = predict_with_bundle(
        request(),
        bundle,
        config(),
    )

    assert response.release_id == "release-1"
    assert [
        result.prediction
        for result in response.predictions
    ] == [
        100.0,
        0.0,
    ]
    assert [
        result.row_index
        for result in response.predictions
    ] == [
        0,
        1,
    ]
    assert all(
        result.horizon_step == 1
        for result in response.predictions
    )

    model_input = (
        bundle.model.predict.call_args.args[0]
    )

    assert list(model_input.columns) == [
        "Store",
        "feature",
    ]
    assert "extra" not in model_input.columns

    build_features.assert_called_once()


def test_prediction_rejects_missing_model_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle()

    monkeypatch.setattr(
        prediction_service,
        "build_forecasting_inference_features",
        MagicMock(
            return_value=pd.DataFrame(
                {
                    "Store": [
                        1,
                        2,
                    ],
                }
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing model columns",
    ):
        predict_with_bundle(
            request(),
            bundle,
            config(),
        )


def test_prediction_rejects_model_without_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle()
    bundle.model.metadata.get_input_schema.return_value = (
        None
    )

    monkeypatch.setattr(
        prediction_service,
        "build_forecasting_inference_features",
        MagicMock(
            return_value=pd.DataFrame(
                {
                    "Store": [
                        1,
                        2,
                    ],
                }
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="no input signature",
    ):
        predict_with_bundle(
            request(),
            bundle,
            config(),
        )


def test_prediction_rejects_wrong_output_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle()
    bundle.model.predict.return_value = np.log1p(
        [
            100.0,
        ]
    )

    monkeypatch.setattr(
        prediction_service,
        "build_forecasting_inference_features",
        MagicMock(
            return_value=pd.DataFrame(
                {
                    "Store": [
                        1,
                        2,
                    ],
                    "feature": [
                        1.0,
                        2.0,
                    ],
                }
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="output length does not match",
    ):
        predict_with_bundle(
            request(),
            bundle,
            config(),
        )


def test_prediction_service_uses_active_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_bundle()
    model_manager = MagicMock()
    model_manager.get_bundle.return_value = bundle
    expected_response = MagicMock()
    predict = MagicMock(
        return_value=expected_response
    )

    monkeypatch.setattr(
        prediction_service,
        "predict_with_bundle",
        predict,
    )

    service = PredictionService(
        model_manager,
        config(),
    )
    result = service.predict(request())

    assert result is expected_response
    model_manager.get_bundle.assert_called_once_with()
    predict.assert_called_once_with(
        request(),
        bundle,
        config(),
    )