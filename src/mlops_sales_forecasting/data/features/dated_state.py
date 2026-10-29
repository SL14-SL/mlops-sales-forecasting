import math

import pandas as pd


def build_dated_feature_state(
    features: pd.DataFrame,
    *,
    history_length: int,
    entity_column: str = "Store",
    date_column: str = "Date",
    target_column: str = "Sales",
) -> dict:
    """Build versioned, dated target history for each forecasting entity."""
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or history_length < 1
    ):
        raise ValueError("History length must be a positive integer.")

    required_columns = [entity_column, date_column, target_column]
    missing_columns = set(required_columns) - set(features.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required state columns: {sorted(missing_columns)}."
        )

    frame = features[required_columns].copy()

    if frame.empty:
        raise ValueError("Forecasting state requires observations.")

    if frame.isna().any().any():
        raise ValueError("Forecasting state contains missing values.")

    entities = pd.to_numeric(frame[entity_column], errors="raise")

    if (
        not entities.map(math.isfinite).all()
        or (entities <= 0).any()
        or (entities % 1 != 0).any()
    ):
        raise ValueError("Store identifiers must be positive integers.")

    frame[entity_column] = entities.astype("int64")

    dates = pd.to_datetime(frame[date_column], errors="raise")

    if dates.isna().any():
        raise ValueError("Forecasting state contains invalid dates.")

    if dates.dt.tz is not None:
        raise ValueError("Observation dates must be timezone-naive.")

    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Observation dates must not contain a time component.")

    frame[date_column] = dates

    values = pd.to_numeric(frame[target_column], errors="raise")

    if not values.map(math.isfinite).all() or (values < 0).any():
        raise ValueError("Sales observations must be finite and non-negative.")

    frame[target_column] = values.astype(float)

    if frame.duplicated([entity_column, date_column]).any():
        raise ValueError("Duplicate store/date observations are not allowed.")

    frame = frame.sort_values([entity_column, date_column])

    stores = {}

    for entity, observations in frame.groupby(entity_column, sort=True):
        recent = observations.tail(history_length)

        stores[str(int(entity))] = [
            {
                "date": observed_date.date().isoformat(),
                "value": float(value),
            }
            for observed_date, value in zip(
                recent[date_column],
                recent[target_column],
                strict=True,
            )
        ]

    return {
        "schema_version": 1,
        "stores": stores,
    }