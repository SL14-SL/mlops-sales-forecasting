from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.configs.loader import file_exists, get_path, load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")

CALENDAR_FEATURE_COLUMNS = [
    "days_until_state_holiday",
    "days_since_state_holiday",
    "is_day_before_state_holiday",
    "is_day_after_state_holiday",
    "days_until_school_holiday_start",
    "days_since_school_holiday_start",
    "days_until_school_holiday_end",
    "days_since_school_holiday_end",
    "is_school_holiday_start",
    "is_school_holiday_end",
]


def _normalize_state_holiday(series: pd.Series) -> pd.Series:
    """Convert StateHoliday values into a boolean holiday indicator."""
    normalized = (
        series.astype(str)
        .str.lower()
        .str.strip()
    )

    no_holiday_values = {
        "0",
        "0.0",
        "",
        "nan",
        "none",
    }

    return ~normalized.isin(no_holiday_values)


def _normalize_binary_indicator(series: pd.Series) -> pd.Series:
    """Convert a numeric or string indicator into a boolean series."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    return numeric.astype(int).eq(1)


def _distance_to_events(
    dates: pd.Series,
    event_mask: pd.Series,
    *,
    direction: str,
    maximum_distance: int,
) -> np.ndarray:
    """Calculate clipped day distances to previous or next events."""
    date_values = (
        pd.to_datetime(dates)
        .to_numpy(dtype="datetime64[D]")
    )

    mask_values = event_mask.to_numpy(dtype=bool)
    event_dates = date_values[mask_values]

    missing_value = maximum_distance + 1
    distances = np.full(
        len(date_values),
        missing_value,
        dtype=int,
    )

    if len(event_dates) == 0:
        return distances

    if direction == "next":
        positions = np.searchsorted(
            event_dates,
            date_values,
            side="left",
        )

        valid = positions < len(event_dates)

        distances[valid] = (
            event_dates[positions[valid]]
            - date_values[valid]
        ).astype("timedelta64[D]").astype(int)

    elif direction == "previous":
        positions = (
            np.searchsorted(
                event_dates,
                date_values,
                side="right",
            )
            - 1
        )

        valid = positions >= 0

        distances[valid] = (
            date_values[valid]
            - event_dates[positions[valid]]
        ).astype("timedelta64[D]").astype(int)

    else:
        raise ValueError(
            f"Unsupported distance direction: {direction}"
        )

    return np.clip(
        distances,
        0,
        missing_value,
    )


def _build_store_calendar(
    store_df: pd.DataFrame,
    *,
    entity_column: str,
    date_column: str,
    maximum_distance: int,
) -> pd.DataFrame:
    """Build known calendar features for one store."""
    store_df = (
        store_df.sort_values(date_column)
        .drop_duplicates(
            subset=[entity_column, date_column],
            keep="last",
        )
        .copy()
    )

    state_holiday = _normalize_state_holiday(
        store_df["StateHoliday"]
    )
    school_holiday = _normalize_binary_indicator(
        store_df["SchoolHoliday"]
    )

    previous_school_holiday = school_holiday.shift(
        1,
        fill_value=False,
    )
    next_school_holiday = school_holiday.shift(
        -1,
        fill_value=False,
    )

    school_holiday_start = (
        school_holiday
        & ~previous_school_holiday
    )
    school_holiday_end = (
        school_holiday
        & ~next_school_holiday
    )

    result = store_df[
        [
            entity_column,
            date_column,
        ]
    ].copy()

    result["days_until_state_holiday"] = _distance_to_events(
        store_df[date_column],
        state_holiday,
        direction="next",
        maximum_distance=maximum_distance,
    )
    result["days_since_state_holiday"] = _distance_to_events(
        store_df[date_column],
        state_holiday,
        direction="previous",
        maximum_distance=maximum_distance,
    )

    result["is_day_before_state_holiday"] = (
        result["days_until_state_holiday"] == 1
    ).astype(int)
    result["is_day_after_state_holiday"] = (
        result["days_since_state_holiday"] == 1
    ).astype(int)

    result[
        "days_until_school_holiday_start"
    ] = _distance_to_events(
        store_df[date_column],
        school_holiday_start,
        direction="next",
        maximum_distance=maximum_distance,
    )
    result[
        "days_since_school_holiday_start"
    ] = _distance_to_events(
        store_df[date_column],
        school_holiday_start,
        direction="previous",
        maximum_distance=maximum_distance,
    )

    result[
        "days_until_school_holiday_end"
    ] = _distance_to_events(
        store_df[date_column],
        school_holiday_end,
        direction="next",
        maximum_distance=maximum_distance,
    )
    result[
        "days_since_school_holiday_end"
    ] = _distance_to_events(
        store_df[date_column],
        school_holiday_end,
        direction="previous",
        maximum_distance=maximum_distance,
    )

    result["is_school_holiday_start"] = (
        school_holiday_start.astype(int).to_numpy()
    )
    result["is_school_holiday_end"] = (
        school_holiday_end.astype(int).to_numpy()
    )

    return result


def build_known_calendar(
    source_df: pd.DataFrame,
    *,
    maximum_distance: int = 30,
) -> pd.DataFrame:
    """
    Build known future calendar features without using target values.

    Only Store, Date, StateHoliday, and SchoolHoliday are consumed.
    """
    data_cfg = TRAIN_CFG["data"]

    entity_column = data_cfg["id_columns"][0]
    date_column = data_cfg["time_column"]

    required_columns = [
        entity_column,
        date_column,
        "StateHoliday",
        "SchoolHoliday",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in source_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Calendar source is missing columns: {missing_columns}"
        )

    calendar_source = source_df[
        required_columns
    ].copy()

    calendar_source[date_column] = pd.to_datetime(
        calendar_source[date_column],
        errors="coerce",
    )
    calendar_source[entity_column] = pd.to_numeric(
        calendar_source[entity_column],
        errors="coerce",
    )

    calendar_source = calendar_source.dropna(
        subset=[
            entity_column,
            date_column,
        ]
    )
    calendar_source[entity_column] = (
        calendar_source[entity_column].astype(int)
    )

    store_calendars = []

    for _, store_df in calendar_source.groupby(
        entity_column,
        sort=False,
    ):
        store_calendar = _build_store_calendar(
            store_df,
            entity_column=entity_column,
            date_column=date_column,
            maximum_distance=maximum_distance,
        )
        store_calendars.append(store_calendar)

    if not store_calendars:
        raise ValueError(
            "Calendar source did not contain usable store data."
        )

    calendar_df = pd.concat(
        store_calendars,
        ignore_index=True,
    )

    calendar_df = calendar_df.sort_values(
        [
            entity_column,
            date_column,
        ]
    ).reset_index(drop=True)

    return calendar_df


def create_known_calendar_artifact(
    *,
    source_path: str | None = None,
    output_path: str | None = None,
    maximum_distance: int = 30,
) -> str:
    """Create and persist the known calendar artifact."""
    raw_path = get_path("raw_data")
    features_path = get_path("features")

    source_path = source_path or f"{raw_path}/train.csv"
    output_path = (
        output_path
        or f"{features_path}/known_calendar.parquet"
    )

    if not file_exists(source_path):
        raise FileNotFoundError(
            f"Calendar source not found: {source_path}"
        )

    logger.info(
        "Building known calendar from: %s",
        source_path,
    )

    source_df = pd.read_csv(
        source_path,
        usecols=[
            "Store",
            "Date",
            "StateHoliday",
            "SchoolHoliday",
        ],
        dtype={
            "StateHoliday": str,
        },
    )

    calendar_df = build_known_calendar(
        source_df,
        maximum_distance=maximum_distance,
    )

    if not output_path.startswith("gs://"):
        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    calendar_df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Known calendar saved | path=%s | rows=%s",
        output_path,
        len(calendar_df),
    )

    return output_path


def load_known_calendar(
    calendar_path: str | None = None,
) -> pd.DataFrame:
    """Load the known calendar artifact."""
    calendar_path = (
        calendar_path
        or f"{get_path('features')}/known_calendar.parquet"
    )

    if not file_exists(calendar_path):
        raise FileNotFoundError(
            f"Known calendar not found: {calendar_path}"
        )

    calendar_df = pd.read_parquet(calendar_path)

    data_cfg = TRAIN_CFG["data"]
    entity_column = data_cfg["id_columns"][0]
    date_column = data_cfg["time_column"]

    calendar_df[entity_column] = pd.to_numeric(
        calendar_df[entity_column],
        errors="raise",
    ).astype(int)

    calendar_df[date_column] = pd.to_datetime(
        calendar_df[date_column],
        errors="raise",
    )

    return calendar_df


def merge_known_calendar_features(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Merge known calendar features by entity and date."""
    if calendar_df is None or calendar_df.empty:
        if strict:
            raise ValueError(
                "Known calendar is missing or empty."
            )
        return df.copy()

    data_cfg = TRAIN_CFG["data"]
    entity_column = data_cfg["id_columns"][0]
    date_column = data_cfg["time_column"]

    result = df.copy()

    result[entity_column] = pd.to_numeric(
        result[entity_column],
        errors="raise",
    ).astype(int)
    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="raise",
    )

    merge_columns = [
        entity_column,
        date_column,
        *CALENDAR_FEATURE_COLUMNS,
    ]

    result = result.merge(
        calendar_df[merge_columns],
        on=[
            entity_column,
            date_column,
        ],
        how="left",
        validate="many_to_one",
    )

    missing_calendar = result[
        CALENDAR_FEATURE_COLUMNS
    ].isna().any(axis=1)

    if strict and missing_calendar.any():
        missing_keys = result.loc[
            missing_calendar,
            [
                entity_column,
                date_column,
            ],
        ].head(5)

        raise ValueError(
            "Known calendar lookup failed for rows: "
            f"{missing_keys.to_dict(orient='records')}"
        )

    if missing_calendar.any():
        result.loc[
            missing_calendar,
            CALENDAR_FEATURE_COLUMNS,
        ] = 31

        for binary_column in [
            "is_day_before_state_holiday",
            "is_day_after_state_holiday",
            "is_school_holiday_start",
            "is_school_holiday_end",
        ]:
            result.loc[
                missing_calendar,
                binary_column,
            ] = 0

    result[CALENDAR_FEATURE_COLUMNS] = result[
        CALENDAR_FEATURE_COLUMNS
    ].astype(int)

    return result