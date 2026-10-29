import math
from datetime import date

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


def _dated_state_to_frame(state: dict) -> pd.DataFrame:
    """Validate the state structure and extract dated observations."""
    if (
        not isinstance(state, dict)
        or set(state) != {"schema_version", "stores"}
        or type(state["schema_version"]) is not int
        or state["schema_version"] != 1
        or not isinstance(state["stores"], dict)
    ):
        raise ValueError("Expected dated forecasting state with schema_version=1.")

    records = []

    for store_id, observations in state["stores"].items():
        if (
            not isinstance(store_id, str)
            or not store_id.isascii()
            or not store_id.isdecimal()
            or int(store_id) <= 0
            or str(int(store_id)) != store_id
        ):
            raise ValueError("State store keys must be canonical positive integers.")

        if not isinstance(observations, list) or not observations:
            raise ValueError("Each store must contain dated observations.")

        for observation in observations:
            if (
                not isinstance(observation, dict)
                or set(observation) != {"date", "value"}
            ):
                raise ValueError("Each observation requires date and value.")

            observed_date = observation["date"]

            if not isinstance(observed_date, str):
                raise ValueError("Observation dates must use YYYY-MM-DD.")

            try:
                parsed_date = date.fromisoformat(observed_date)
            except ValueError as exc:
                raise ValueError("Observation dates must use YYYY-MM-DD.") from exc

            if parsed_date.isoformat() != observed_date:
                raise ValueError("Observation dates must use YYYY-MM-DD.")

            value = observation["value"]

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Observation values must be numeric.")

            records.append(
                {
                    "Store": int(store_id),
                    "Date": observed_date,
                    "Sales": value,
                }
            )

    return pd.DataFrame(records, columns=["Store", "Date", "Sales"])


def merge_dated_feature_state(
    state: dict,
    batch: pd.DataFrame,
    *,
    history_length: int,
    entity_column: str = "Store",
    date_column: str = "Date",
    target_column: str = "Sales",
) -> dict:
    """Merge ground truth by store/date; incoming observations replace old values."""
    existing = _dated_state_to_frame(state)

    if not existing.empty:
        # Validate the entire existing history before combining observations.
        build_dated_feature_state(
            existing,
            history_length=len(existing),
        )

    incoming_state = build_dated_feature_state(
        batch,
        history_length=max(len(batch), 1),
        entity_column=entity_column,
        date_column=date_column,
        target_column=target_column,
    )
    incoming = _dated_state_to_frame(incoming_state)

    combined = pd.concat([existing, incoming], ignore_index=True)

    # Incoming values win when correcting an already observed store/date.
    combined = combined.drop_duplicates(
        subset=["Store", "Date"],
        keep="last",
    )

    return build_dated_feature_state(
        combined,
        history_length=history_length,
    )


def build_dated_lag_features(
    inputs: pd.DataFrame,
    state: dict,
    *,
    lags: list[int],
    rolling_windows: list[int],
    entity_column: str = "Store",
    date_column: str = "Date",
    target_column: str = "Sales",
) -> pd.DataFrame:
    """Build calendar-based lag features for next-day forecasts."""
    settings = [*lags, *rolling_windows]

    if not settings or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in settings
    ):
        raise ValueError("Lag offsets and windows must be positive integers.")

    required_columns = {entity_column, date_column}

    if not required_columns.issubset(inputs.columns):
        raise ValueError("Forecast inputs require entity and date columns.")

    if inputs.empty:
        raise ValueError("Forecast inputs must not be empty.")

    history = _dated_state_to_frame(state)

    if history.empty:
        raise ValueError("Forecasting state has no observations.")

    # Validate values and duplicate store/date keys before using the state.
    build_dated_feature_state(history, history_length=len(history))
    history["Date"] = pd.to_datetime(history["Date"])

    histories = {
        int(store): dict(zip(group["Date"], group["Sales"], strict=True))
        for store, group in history.groupby("Store")
    }

    entities = pd.to_numeric(inputs[entity_column], errors="raise")

    if (
        entities.isna().any()
        or not entities.map(math.isfinite).all()
        or (entities <= 0).any()
        or (entities % 1 != 0).any()
    ):
        raise ValueError("Store identifiers must be positive integers.")

    dates = pd.to_datetime(inputs[date_column], errors="raise")

    if (
        dates.isna().any()
        or dates.dt.tz is not None
        or not dates.eq(dates.dt.normalize()).all()
    ):
        raise ValueError("Forecast dates must be timezone-naive calendar dates.")

    required_offsets = set(lags)

    for window in rolling_windows:
        required_offsets.update(range(1, window + 1))

    prefix = target_column.strip().lower().replace(" ", "_")
    records = []

    for store, forecast_date in zip(entities, dates, strict=True):
        store_id = int(store)
        observations = histories.get(store_id)

        if not observations:
            raise ValueError(f"No dated history for Store={store_id}.")

        next_date = max(observations) + pd.Timedelta(days=1)

        if forecast_date != next_date:
            raise ValueError(
                f"Store={store_id} supports only next-day forecasts: "
                f"expected {next_date.date()}, received {forecast_date.date()}."
            )

        missing_dates = [
            (forecast_date - pd.Timedelta(days=offset)).date().isoformat()
            for offset in sorted(required_offsets)
            if forecast_date - pd.Timedelta(days=offset) not in observations
        ]

        if missing_dates:
            raise ValueError(
                f"Missing dated history for Store={store_id}: {missing_dates}."
            )

        record = {
            f"{prefix}_lag_{lag}": float(
                observations[forecast_date - pd.Timedelta(days=lag)]
            )
            for lag in lags
        }

        for window in rolling_windows:
            record[f"{prefix}_rolling_mean_{window}"] = sum(
                observations[forecast_date - pd.Timedelta(days=offset)]
                for offset in range(1, window + 1)
            ) / window

        records.append(record)

    return pd.DataFrame(records, index=inputs.index)