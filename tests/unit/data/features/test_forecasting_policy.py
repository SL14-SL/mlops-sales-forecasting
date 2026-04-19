import pandas as pd

from src.data.features.forecasting_policy import (
    add_competition_duration_features,
    add_promo_duration_features,
)






def test_add_competition_duration_features():
    df = pd.DataFrame(
        {
            "year": [2026],
            "month": [3],
            "CompetitionOpenSinceYear": [2025],
            "CompetitionOpenSinceMonth": [1],
        }
    )

    result = add_competition_duration_features(df)

    assert "CompetitionOpen" in result.columns
    assert result.loc[0, "CompetitionOpen"] == 14


def test_add_promo_duration_features():
    df = pd.DataFrame(
        {
            "year": [2026],
            "WeekOfYear": [10],
            "Promo2SinceYear": [2025],
            "Promo2SinceWeek": [2],
        }
    )

    result = add_promo_duration_features(df)

    assert "PromoOpen" in result.columns
    assert result.loc[0, "PromoOpen"] > 0