import pandas as pd

from mlops_sales_forecasting.utils.logger import get_logger

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
        df["CompetitionOpen"] = 12 * (df["year"] - df["CompetitionOpenSinceYear"]) + (
            df["month"] - df["CompetitionOpenSinceMonth"]
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

    if promo_column in df.columns:
        calendar_columns = {"planned_promo", "days_since_promo_start"}

        if not calendar_columns.issubset(df.columns):
            raise ValueError(
                "Promo duration requires release-bound promo calendar features."
            )

        promo = pd.to_numeric(df[promo_column], errors="raise")
        planned = pd.to_numeric(df["planned_promo"], errors="raise")

        if not promo.isin([0, 1]).all() or not planned.isin([0, 1]).all():
            raise ValueError("Promo flags must be zero or one.")

        if not promo.eq(planned).all():
            raise ValueError(
                "Promo does not match the known promotion calendar."
            )

        duration = pd.to_numeric(
            df["days_since_promo_start"],
            errors="raise",
        )

        if not duration.ge(-1).all() or not duration.mod(1).eq(0).all():
            raise ValueError(
                "Promo duration must be an integer greater than or equal to -1."
            )

        if not duration.loc[planned.eq(0)].eq(0).all():
            raise ValueError(
                "Non-promotional dates must have zero promo duration."
            )

        df["days_since_promo_start"] = duration.astype("int64")
    elif "days_since_promo_start" not in df.columns:
        df["days_since_promo_start"] = 0

    return df.drop(
        columns=["planned_promo", "promo_change", "promo_group"],
        errors="ignore",
    )
