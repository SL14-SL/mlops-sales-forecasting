import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error


def align_features_for_evaluation(
    model,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Align evaluation features to the schema expected by a model."""
    booster = model.get_booster()
    expected_features = booster.feature_names or []

    if not expected_features:
        return features

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Evaluation data is missing model features: "
            f"{missing_features}"
        )

    return features[expected_features]


def calculate_promotion_metrics(
    *,
    y_true: pd.Series,
    predictions: np.ndarray,
    evaluation_frame: pd.DataFrame,
) -> tuple[
    dict[str, float],
    dict[str, int],
]:
    """
    Calculate overall and business-segment metrics for promotion.

    Bias convention:
        mean(prediction - actual)

    Negative bias means underprediction.
    Positive bias means overprediction.
    """
    actual = np.asarray(
        y_true,
        dtype=float,
    )
    predicted = np.asarray(
        predictions,
        dtype=float,
    )

    if len(actual) != len(predicted):
        raise ValueError(
            "Prediction and target lengths "
            "do not match."
        )

    if "Promo" not in evaluation_frame.columns:
        raise ValueError(
            "Validation data is missing "
            "required segment column 'Promo'."
        )

    promo_values = pd.to_numeric(
        evaluation_frame["Promo"],
        errors="coerce",
    )

    if promo_values.isna().any():
        raise ValueError(
            "Validation segment column "
            "'Promo' contains invalid values."
        )

    promo_mask = (
        promo_values.to_numpy()
        == 1
    )
    non_promo_mask = (
        promo_values.to_numpy()
        == 0
    )

    def rmse_for_mask(
        mask: np.ndarray,
        segment_name: str,
    ) -> float:
        """
        Calculate RMSE for the rows selected by a Boolean mask.

        Returns:
            The subset RMSE and number of selected observations.
        """
        row_count = int(
            mask.sum()
        )

        if row_count == 0:
            raise ValueError(
                "No validation rows available "
                f"for segment '{segment_name}'."
            )

        return float(
            np.sqrt(
                mean_squared_error(
                    actual[mask],
                    predicted[mask],
                )
            )
        )

    metrics = {
        "overall_rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "promo_rmse": rmse_for_mask(
            promo_mask,
            "promo",
        ),
        "non_promo_rmse": rmse_for_mask(
            non_promo_mask,
            "non_promo",
        ),
        "overall_bias": float(
            np.mean(
                predicted - actual
            )
        ),
    }

    segment_rows = {
        "promo": int(
            promo_mask.sum()
        ),
        "non_promo": int(
            non_promo_mask.sum()
        ),
    }

    return metrics, segment_rows
