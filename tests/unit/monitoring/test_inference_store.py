from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from mlops_sales_forecasting.monitoring.inference_store import (
    build_inference_log_path,
    build_inference_records,
    record_inference_batch,
)


def validated_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "Date": pd.to_datetime(
                [
                    "2026-09-27",
                    "2026-09-28",
                ]
            ),
            "Open": [
                1,
                1,
            ],
            "Promo": [
                0,
                1,
            ],
            "StateHoliday": [
                "0",
                "0",
            ],
        }
    )


def inference_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "Promo": [
                0,
                1,
            ],
            "StoreType": [
                "a",
                "b",
            ],
            "CompetitionDistance": [
                500.0,
                1500.0,
            ],
            "private_technical_value": [
                100,
                200,
            ],
        }
    )


def test_build_inference_records_uses_allowlist() -> None:
    observed_at = datetime(
        2026,
        9,
        27,
        5,
        0,
        tzinfo=UTC,
    )

    result = build_inference_records(
        validated_input=validated_input(),
        inference_features=inference_features(),
        predictions=[
            100.0,
            200.0,
        ],
        release_id="release-7",
        request_id="request-123",
        feature_allowlist=[
            "Promo",
            "StoreType",
            "CompetitionDistance",
        ],
        observed_at=observed_at,
    )

    assert list(result.columns) == [
        "Store",
        "Date",
        "Promo",
        "StoreType",
        "CompetitionDistance",
        "prediction",
        "release_id",
        "request_id",
        "row_index",
        "timestamp",
    ]
    assert "private_technical_value" not in result.columns
    assert result["prediction"].tolist() == [
        100.0,
        200.0,
    ]
    assert result["release_id"].unique().tolist() == [
        "release-7",
    ]


def test_build_inference_records_uses_input_feature_fallback() -> None:
    result = build_inference_records(
        validated_input=validated_input(),
        inference_features=inference_features(),
        predictions=[
            100.0,
            200.0,
        ],
        release_id="release-7",
        request_id="request-123",
        feature_allowlist=[
            "StateHoliday",
        ],
    )

    assert result["StateHoliday"].tolist() == [
        "0",
        "0",
    ]


def test_build_inference_records_rejects_wrong_length() -> None:
    with pytest.raises(
        ValueError,
        match="Prediction count",
    ):
        build_inference_records(
            validated_input=validated_input(),
            inference_features=inference_features(),
            predictions=[
                100.0,
            ],
            release_id="release-7",
            request_id="request-123",
            feature_allowlist=[],
        )


def test_build_inference_log_path() -> None:
    result = build_inference_log_path(
        predictions_path="data/predictions",
        observed_at=datetime(
            2026,
            9,
            27,
            23,
            0,
            tzinfo=UTC,
        ),
        file_id="request-123",
    )

    assert result == ("data/predictions/history/date=2026-09-27/request-123.parquet")


def test_record_inference_batch(
    tmp_path: Path,
) -> None:
    observed_at = datetime(
        2026,
        9,
        27,
        5,
        0,
        tzinfo=UTC,
    )

    output_path = record_inference_batch(
        validated_input=validated_input(),
        inference_features=inference_features(),
        predictions=[
            100.0,
            200.0,
        ],
        release_id="release-7",
        request_id="request-123",
        feature_allowlist=[
            "Promo",
            "StoreType",
        ],
        predictions_path=str(tmp_path / "predictions"),
        observed_at=observed_at,
        file_id="batch-123",
    )

    assert output_path.endswith("history/date=2026-09-27/batch-123.parquet")

    persisted = pd.read_parquet(output_path)

    assert len(persisted) == 2
    assert persisted["request_id"].tolist() == [
        "request-123",
        "request-123",
    ]
    assert persisted["prediction"].tolist() == [
        100.0,
        200.0,
    ]
