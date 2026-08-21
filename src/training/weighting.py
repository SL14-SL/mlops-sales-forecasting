import numpy as np
import pandas as pd



def build_recency_weights(
    dates: pd.Series,
    promo_values: pd.Series,
    weighting_config: dict,
) -> np.ndarray:
    """
    Assign higher sample weights to recent promotional observations.
    """
    parsed_dates = pd.to_datetime(
        dates,
        errors="raise",
    )

    if parsed_dates.isna().any():
        raise ValueError(
            "Training dates contain missing values."
        )

    parsed_promo = (
        pd.to_numeric(
            promo_values,
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
    )

    latest_training_date = parsed_dates.max()

    age_days = (
        latest_training_date - parsed_dates
    ).dt.days

    recent_30_day_promo = (
        parsed_promo
        & age_days.le(30)
    )

    recent_60_day_promo = (
        parsed_promo
        & age_days.gt(30)
        & age_days.le(60)
    )

    recent_120_day_promo = (
        parsed_promo
        & age_days.gt(60)
        & age_days.le(120)
    )

    weights = np.select(
        [
            recent_30_day_promo,
            recent_60_day_promo,
            recent_120_day_promo,
        ],
        [
            float(
                weighting_config.get(
                    "last_30_days_weight",
                    10.0,
                )
            ),
            float(
                weighting_config.get(
                    "last_60_days_weight",
                    5.0,
                )
            ),
            float(
                weighting_config.get(
                    "last_120_days_weight",
                    2.0,
                )
            ),
        ],
        default=float(
            weighting_config.get(
                "default_weight",
                1.0,
            )
        ),
    )

    return weights.astype(np.float32)