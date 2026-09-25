import numpy as np
import pandas as pd
import pytest

from mlops_sales_forecasting.training.evaluate_metrics import (
    align_features_for_evaluation,
    calculate_promotion_metrics,
)


class FakeBooster:
    def __init__(
        self,
        feature_names: list[str],
    ) -> None:
        self.feature_names = feature_names


class FakeModel:
    def __init__(
        self,
        feature_names: list[str],
    ) -> None:
        self.feature_names = feature_names

    def get_booster(
        self,
    ) -> FakeBooster:
        return FakeBooster(self.feature_names)


def test_align_features_removes_extra_columns() -> None:
    model = FakeModel(
        [
            "Store",
            "Promo",
        ]
    )
    features = pd.DataFrame(
        {
            "Store": [
                1,
            ],
            "calendar_feature": [
                3,
            ],
            "Promo": [
                1,
            ],
        }
    )

    result = align_features_for_evaluation(
        model,
        features,
    )

    assert result.columns.tolist() == [
        "Store",
        "Promo",
    ]


def test_align_features_rejects_missing_columns() -> None:
    model = FakeModel(
        [
            "Store",
            "Promo",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Promo",
    ):
        align_features_for_evaluation(
            model,
            pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
        )


def test_calculate_metrics_by_promo_segment() -> None:
    metrics, segment_rows = calculate_promotion_metrics(
        y_true=pd.Series(
            [
                100.0,
                200.0,
                100.0,
                200.0,
            ]
        ),
        predictions=np.array(
            [
                110.0,
                190.0,
                130.0,
                170.0,
            ]
        ),
        evaluation_frame=pd.DataFrame(
            {
                "Promo": [
                    1,
                    1,
                    0,
                    0,
                ],
            }
        ),
    )

    assert segment_rows == {
        "promo": 2,
        "non_promo": 2,
    }
    assert metrics["promo_rmse"] == pytest.approx(10.0)
    assert metrics["non_promo_rmse"] == pytest.approx(30.0)
    assert metrics["overall_bias"] == pytest.approx(0.0)


def test_calculate_metrics_requires_promo() -> None:
    with pytest.raises(
        ValueError,
        match="required segment column",
    ):
        calculate_promotion_metrics(
            y_true=pd.Series(
                [
                    100.0,
                ]
            ),
            predictions=np.array(
                [
                    100.0,
                ]
            ),
            evaluation_frame=pd.DataFrame(
                {
                    "other": [
                        1,
                    ],
                }
            ),
        )
