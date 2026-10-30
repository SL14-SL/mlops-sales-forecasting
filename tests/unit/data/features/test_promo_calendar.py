import pandas as pd
import pytest

from mlops_sales_forecasting.data.features.promo_calendar import (
    build_promo_calendar,
)


def test_promo_duration_tracks_transitions_per_store() -> None:
    source = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1, 1, 2, 2],
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
                "2026-01-01",
                "2026-01-02",
            ],
            "Promo": [0, 1, 1, 0, 1, 0, 1],
        }
    )

    result = build_promo_calendar(source.iloc[::-1])

    assert result["days_since_promo_start"].tolist() == [
        0, 0, 1, 0, 0, 0, 0,
    ]
    assert result["planned_promo"].tolist() == [0, 1, 1, 0, 1, 0, 1]
    assert source["Date"].iloc[0] == "2026-01-01"


def test_unknown_start_and_calendar_gaps_are_not_invented() -> None:
    source = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1, 1],
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-04",
                "2026-01-05",
                "2026-01-06",
            ],
            "Promo": [1, 1, 1, 0, 1],
        }
    )

    result = build_promo_calendar(source)

    assert result["days_since_promo_start"].tolist() == [-1, -1, -1, 0, 0]


def test_future_schedule_does_not_change_past_features() -> None:
    source = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1],
            "Date": pd.date_range("2026-01-01", periods=4),
            "Promo": [0, 1, 1, 0],
            "Sales": [100, 200, 300, 400],
        }
    )

    prefix = build_promo_calendar(source.iloc[:3])
    full = build_promo_calendar(source)

    pd.testing.assert_frame_equal(
        prefix,
        full.iloc[:3].reset_index(drop=True),
    )
    assert "Sales" not in full.columns


def test_rejects_duplicate_store_dates() -> None:
    source = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2026-01-01", "2026-01-01"],
            "Promo": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="Duplicate store/date"):
        build_promo_calendar(source)


@pytest.mark.parametrize("promo", [-1, 2, None])
def test_rejects_invalid_promo_flags(promo) -> None:
    source = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2026-01-01"],
            "Promo": [promo],
        }
    )

    with pytest.raises(ValueError):
        build_promo_calendar(source)