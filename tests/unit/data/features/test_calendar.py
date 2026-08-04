import pandas as pd

from src.data.features.calendar import (
    build_known_calendar,
    merge_known_calendar_features,
    prepare_known_calendar_lookup,
)


def build_calendar_source() -> pd.DataFrame:
    """Create a small deterministic calendar fixture."""
    dates = pd.date_range(
        "2015-04-01",
        "2015-04-08",
        freq="D",
    )

    return pd.DataFrame(
        {
            "Store": [1] * len(dates),
            "Date": dates,
            "StateHoliday": [
                "0",
                "0",
                "a",
                "0",
                "0",
                "a",
                "0",
                "0",
            ],
            "SchoolHoliday": [
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
            ],
        }
    )


def test_build_known_calendar_adds_state_holiday_distances():
    source_df = build_calendar_source()

    result = build_known_calendar(
        source_df,
        maximum_distance=30,
    )

    april_second = result.loc[
        result["Date"] == pd.Timestamp("2015-04-02")
    ].iloc[0]

    assert april_second["days_until_state_holiday"] == 1
    assert april_second["is_day_before_state_holiday"] == 1

    april_fourth = result.loc[
        result["Date"] == pd.Timestamp("2015-04-04")
    ].iloc[0]

    assert april_fourth["days_since_state_holiday"] == 1
    assert april_fourth["is_day_after_state_holiday"] == 1
    assert april_fourth["days_until_state_holiday"] == 2


def test_build_known_calendar_adds_school_holiday_boundaries():
    source_df = build_calendar_source()

    result = build_known_calendar(
        source_df,
        maximum_distance=30,
    )

    april_second = result.loc[
        result["Date"] == pd.Timestamp("2015-04-02")
    ].iloc[0]

    assert april_second["is_school_holiday_start"] == 1
    assert april_second["days_until_school_holiday_start"] == 0

    april_fourth = result.loc[
        result["Date"] == pd.Timestamp("2015-04-04")
    ].iloc[0]

    assert april_fourth["is_school_holiday_end"] == 1
    assert april_fourth["days_since_school_holiday_start"] == 2


def test_merge_known_calendar_features_uses_store_and_date():
    source_df = build_calendar_source()
    calendar_df = build_known_calendar(source_df)

    request_df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2015-04-02"],
            "Promo": [1],
        }
    )

    result = merge_known_calendar_features(
        request_df,
        calendar_df,
        strict=True,
    )

    assert result.loc[0, "days_until_state_holiday"] == 1
    assert result.loc[0, "is_day_before_state_holiday"] == 1

def test_merge_known_calendar_features_supports_indexed_lookup():
    source_df = build_calendar_source()
    calendar_df = build_known_calendar(source_df)

    indexed_calendar = prepare_known_calendar_lookup(
        calendar_df
    )

    request_df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2015-04-02"],
            "Promo": [1],
        }
    )

    result = merge_known_calendar_features(
        request_df,
        indexed_calendar,
        strict=True,
    )

    assert result.loc[
        0,
        "days_until_state_holiday",
    ] == 1

    assert result.loc[
        0,
        "is_day_before_state_holiday",
    ] == 1