import pandas as pd
import pytest

from src.deployment.prediction_probe import (
    build_prediction_probe,
)


def test_builds_probe_from_newest_usable_row(
    tmp_path,
):
    source_path = (
        tmp_path
        / "validated_train.parquet"
    )

    pd.DataFrame(
        [
            {
                "Store": 2,
                "DayOfWeek": 4,
                "Date": "2015-04-23",
                "Customers": 400,
                "Open": 1,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            },
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Date": "2015-04-24",
                "Customers": 500,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            },
        ]
    ).to_parquet(
        source_path,
        index=False,
    )

    result = build_prediction_probe(
        validated_data_path=str(
            source_path
        ),
    )

    assert result["inputs"] == [
        {
            "Store": 1,
            "DayOfWeek": 5,
            "Date": "2015-04-24",
            "Customers": 500,
            "Open": 1,
            "Promo": 1,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }
    ]

    assert result["context"]["purpose"] == (
        "post_deployment_verification"
    )


def test_rejects_missing_probe_columns(
    tmp_path,
):
    source_path = tmp_path / "invalid.parquet"

    pd.DataFrame(
        [
            {
                "Store": 1,
                "Date": "2015-04-24",
            }
        ]
    ).to_parquet(
        source_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        build_prediction_probe(
            validated_data_path=str(
                source_path
            ),
        )


def test_rejects_source_without_usable_row(
    tmp_path,
):
    source_path = tmp_path / "closed.parquet"

    pd.DataFrame(
        [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Date": "2015-04-24",
                "Customers": 0,
                "Open": 0,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            }
        ]
    ).to_parquet(
        source_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="no usable open-store row",
    ):
        build_prediction_probe(
            validated_data_path=str(
                source_path
            ),
        )