import pandas as pd

from src.data.features.common import add_basic_calendar_features
from src.utils.logger import get_logger


logger = get_logger(__name__)


def normalize_feature_prefix(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def get_lag_feature_names(
    target_column: str,
    *,
    lags: list[int],
    rolling_windows: list[int],
) -> dict[str, str]:
    """
    Build deterministic feature names for configured lags and rolling windows.

    Returns:
        A mapping from logical lag definitions to dataframe column names.
    """
    prefix = normalize_feature_prefix(target_column)

    names = {}

    for lag in lags:
        names[f"lag_{lag}"] = f"{prefix}_lag_{lag}"

    for window in rolling_windows:
        names[f"rolling_mean_{window}"] = f"{prefix}_rolling_mean_{window}"

    return names


def sort_frame(
    df: pd.DataFrame,
    *,
    entity_column: str,
    date_column: str,
) -> pd.DataFrame:
    """
    Return a copy sorted by entity and date when both columns are present.
    """
    df = df.copy()

    if entity_column in df.columns and date_column in df.columns:
        df = df.sort_values(by=[entity_column, date_column])

    return df


def add_temporal_features(
    df: pd.DataFrame,
    *,
    date_column: str,
) -> pd.DataFrame:
    return add_basic_calendar_features(df, date_column=date_column)


def add_training_lag_features(
    df: pd.DataFrame,
    *,
    entity_column: str,
    target_column: str,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add leakage-safe lag and rolling-mean features for model training.

    Lag values are calculated independently per entity. Rolling means use only
    previous target observations by shifting the target before window aggregation.
    Unavailable historical values are filled with zero.

    Args:
        df: Chronologically sortable training observations.
        entity_column: Column identifying an independent forecasting series.
        target_column: Column from which historical features are calculated.
        lags: Historical offsets to create.
        rolling_windows: Window sizes for shifted rolling means.

    Returns:
        A copy containing the generated lag and rolling features.
    """
    df = df.copy()

    if target_column not in df.columns:
        return df

    lags = lags or [1, 7]
    rolling_windows = rolling_windows or [7]

    logger.info(
        f"Training mode: calculating lag features from target column "
        f"(lags={lags}, rolling_windows={rolling_windows})."
    )

    feature_names = get_lag_feature_names(
        target_column,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    created_cols = []

    for lag in lags:
        col_name = feature_names[f"lag_{lag}"]
        df[col_name] = df.groupby(entity_column)[target_column].shift(lag)
        created_cols.append(col_name)

    for window in rolling_windows:
        col_name = feature_names[f"rolling_mean_{window}"]
        df[col_name] = df.groupby(entity_column)[target_column].transform(
            lambda x: x.shift(1).rolling(window=window).mean()
        )
        created_cols.append(col_name)

    df[created_cols] = df[created_cols].fillna(0)

    return df


def initialize_inference_lag_placeholders(
    df: pd.DataFrame,
    *,
    target_column: str,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Initialize missing lag and rolling-feature columns for inference.

    The placeholder values are expected to be replaced with persisted forecasting
    state before model execution.

    Returns:
        A copy containing every configured historical feature column.
    """
    df = df.copy()

    lags = lags or [1, 7]
    rolling_windows = rolling_windows or [7]

    feature_names = get_lag_feature_names(
        target_column,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    for col_name in feature_names.values():
        if col_name not in df.columns:
            df[col_name] = 0.0

    return df