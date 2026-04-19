import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


FORECASTING_TECHNICAL_DROP_COLUMNS = [
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
]


def add_competition_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasting/domain-specific feature:
    Add competition duration when the required source columns exist.
    """
    df = df.copy()

    required = ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth", "year", "month"]
    if all(col in df.columns for col in required):
        logger.info("Calculating competition activity duration.")
        df["CompetitionOpen"] = (
            12 * (df["year"] - df["CompetitionOpenSinceYear"])
            + (df["month"] - df["CompetitionOpenSinceMonth"])
        )
        df["CompetitionOpen"] = df["CompetitionOpen"].apply(lambda x: x if x > 0 else 0)

    return df


def add_promo_duration_features(
        df: pd.DataFrame,
        *,
        entity_column: str = "Store",
        date_column: str = "Date",
        promo_column: str = "Promo",
    ) -> pd.DataFrame:
    """
    Forecasting/domain-specific feature:
    Add promo duration when the required source columns exist.
    """
        
    df = df.copy()

    required = ["Promo2SinceYear", "Promo2SinceWeek", "year", "WeekOfYear"]
    if all(col in df.columns for col in required):
        logger.info("Calculating promo activity duration.")
        df["PromoOpen"] = (
            12 * (df["year"] - df["Promo2SinceYear"])
            + (df["WeekOfYear"] - df["Promo2SinceWeek"]) / 4.0
        )
        df["PromoOpen"] = df["PromoOpen"].clip(lower=0)
    else:
        if "PromoOpen" not in df.columns:
            df["PromoOpen"] = 0

    sequence_required = [entity_column, date_column, promo_column]
    if all(col in df.columns for col in sequence_required):
        df = df.sort_values([entity_column, date_column])

        df["promo_change"] = df[promo_column].ne(
            df.groupby(entity_column)[promo_column].shift()
        )
        df["promo_group"] = df.groupby(entity_column)["promo_change"].cumsum()
        df["days_since_promo_start"] = (
            df.groupby([entity_column, "promo_group"]).cumcount()
        )
    else:
        if "days_since_promo_start" not in df.columns:
            df["days_since_promo_start"] = 0

    return df