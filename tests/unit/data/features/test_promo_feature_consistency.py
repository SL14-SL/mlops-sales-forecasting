import pandas as pd
import pytest

from mlops_sales_forecasting.data.features.build_features import preprocess_data
from mlops_sales_forecasting.data.features.calendar import (
    build_known_calendar,
    merge_known_calendar_features,
    prepare_known_calendar_lookup,
)


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1] * 5,
            "Date": pd.date_range("2026-01-01", periods=5),
            "Promo": [0, 1, 1, 1, 0],
            "StateHoliday": ["0"] * 5,
            "SchoolHoliday": [0] * 5,
            "Sales": [10, 20, 30, 40, 50],
        }
    )


def feature_config() -> dict:
    return {
        "data": {
            "id_columns": ["Store"],
            "target_column": "Sales",
            "time_column": "Date",
        },
        "features": {
            "enabled_steps": ["sort", "promo"],
        },
    }


@pytest.mark.parametrize("indexed", [False, True])
def test_promo_features_match_training_single_and_batch(indexed: bool) -> None:
    source = source_frame()
    calendar = build_known_calendar(source)

    if indexed:
        calendar = prepare_known_calendar_lookup(calendar)

    def process(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
        merged = merge_known_calendar_features(frame, calendar)

        return preprocess_data(
            merged,
            config=feature_config(),
            mode=mode,
        )

    training = process(source, "train")
    batch = process(source.iloc[[3, 1]].drop(columns="Sales"), "inference")
    single = process(source.iloc[[3]].drop(columns="Sales"), "inference")

    target_date = pd.Timestamp("2026-01-04")

    assert training.loc[
        training["Date"].eq(target_date),
        "days_since_promo_start",
    ].tolist() == [2]

    assert batch.loc[
        batch["Date"].eq(target_date),
        "days_since_promo_start",
    ].tolist() == [2]

    assert single["days_since_promo_start"].tolist() == [2]

    for frame in (training, batch, single):
        assert not {
            "planned_promo",
            "promo_change",
            "promo_group",
        }.intersection(frame.columns)


def test_rejects_request_that_conflicts_with_promo_plan() -> None:
    source = source_frame()
    calendar = build_known_calendar(source)
    request = source.iloc[[3]].drop(columns="Sales").assign(Promo=0)
    merged = merge_known_calendar_features(request, calendar)

    with pytest.raises(ValueError, match="Promo does not match"):
        preprocess_data(
            merged,
            config=feature_config(),
            mode="inference",
        )


def test_rejects_calendar_without_promo_features() -> None:
    source = source_frame()
    calendar = build_known_calendar(source).drop(columns="planned_promo")

    with pytest.raises(ValueError, match="missing features"):
        merge_known_calendar_features(source, calendar)