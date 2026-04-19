import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def ensure_datetime_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Ensure a column is parsed as datetime if present.
    """
    df = df.copy()

    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name])

    return df


def add_basic_calendar_features(
    df: pd.DataFrame,
    *,
    date_column: str = "Date",
) -> pd.DataFrame:
    """
    Add generic calendar-based features derived from a datetime column.
    """
    df = df.copy()

    if date_column not in df.columns:
        return df

    df = ensure_datetime_column(df, date_column)

    df["DayOfWeek"] = df[date_column].dt.dayofweek + 1
    df["WeekOfYear"] = df[date_column].dt.isocalendar().week.astype(int)
    df["day"] = df[date_column].dt.day
    df["month"] = df[date_column].dt.month
    df["year"] = df[date_column].dt.year
    df["is_month_start"] = df[date_column].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_column].dt.is_month_end.astype(int)
    df["day_of_month"] = df[date_column].dt.day

    return df


def cast_object_columns_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns to pandas category dtype for model compatibility.
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        logger.info(f"Converting categorical columns: {cat_cols}")
        for col in cat_cols:
            df[col] = df[col].astype(str).astype("category")

    return df


def drop_columns_if_present(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop columns if they exist in the dataframe.
    """
    df = df.copy()

    existing_drops = [col for col in columns if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    return df