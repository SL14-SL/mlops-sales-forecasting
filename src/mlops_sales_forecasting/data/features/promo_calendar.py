from datetime import timedelta

import pandas as pd


def build_promo_calendar(
    source: pd.DataFrame,
    *,
    entity_column: str = "Store",
    date_column: str = "Date",
    promo_column: str = "Promo",
) -> pd.DataFrame:
    """Build promo duration from the dated promotion schedule only."""
    required = [entity_column, date_column, promo_column]
    missing = set(required) - set(source.columns)

    if missing:
        raise ValueError(f"Promo calendar is missing columns: {sorted(missing)}.")

    frame = source[required].copy()

    if frame.empty or frame.isna().any().any():
        raise ValueError("Promo calendar requires non-missing observations.")

    entities = pd.to_numeric(frame[entity_column], errors="raise")

    if (
        not entities.gt(0).all()
        or not entities.mod(1).eq(0).all()
    ):
        raise ValueError("Store identifiers must be positive integers.")

    frame[entity_column] = entities.astype("int64")

    dates = pd.to_datetime(frame[date_column], errors="raise")

    if (
        dates.isna().any()
        or dates.dt.tz is not None
        or not dates.eq(dates.dt.normalize()).all()
    ):
        raise ValueError("Promo dates must be timezone-naive calendar dates.")

    frame[date_column] = dates

    promos = pd.to_numeric(frame[promo_column], errors="raise")

    if not promos.isin([0, 1]).all():
        raise ValueError("Promo flags must be zero or one.")

    frame[promo_column] = promos.astype(int)

    if frame.duplicated([entity_column, date_column]).any():
        raise ValueError("Duplicate store/date observations in promo calendar.")

    frame = frame.sort_values(
        [entity_column, date_column]
    ).reset_index(drop=True)

    previous_store = None
    previous_date = None
    previous_promo = None
    run_start = None
    durations = []

    for store, timestamp, promo in frame[required].itertuples(
        index=False,
        name=None,
    ):
        observed_date = timestamp.date()
        contiguous = (
            store == previous_store
            and previous_date is not None
            and observed_date == previous_date + timedelta(days=1)
        )

        if promo == 0:
            run_start = None
            duration = 0
        else:
            if not contiguous:
                run_start = None
            elif previous_promo == 0:
                run_start = observed_date

            duration = (
                (observed_date - run_start).days
                if run_start is not None
                else -1
            )

        durations.append(duration)
        previous_store = store
        previous_date = observed_date
        previous_promo = promo

    result = frame.rename(columns={promo_column: "planned_promo"})
    result["days_since_promo_start"] = durations

    return result