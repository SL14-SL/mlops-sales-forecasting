import pandas as pd

from src.training.evaluate import (
    align_features_for_evaluation,
)


class FakeBooster:
    def __init__(self, feature_names):
        self.feature_names = feature_names


class FakeModel:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def get_booster(self):
        return FakeBooster(
            self.feature_names
        )


def test_align_features_for_evaluation_removes_extra_columns():
    model = FakeModel(
        [
            "Store",
            "Promo",
        ]
    )

    features = pd.DataFrame(
        {
            "Store": [1],
            "calendar_feature": [3],
            "Promo": [1],
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


def test_align_features_for_evaluation_rejects_missing_columns():
    model = FakeModel(
        [
            "Store",
            "Promo",
        ]
    )

    features = pd.DataFrame(
        {
            "Store": [1],
        }
    )

    try:
        align_features_for_evaluation(
            model,
            features,
        )
    except ValueError as error:
        assert "Promo" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )