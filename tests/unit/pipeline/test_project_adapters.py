from unittest.mock import MagicMock

import pandas as pd

from mlops_sales_forecasting.data.contracts import (
    DatasetCollection,
)
from mlops_sales_forecasting.pipeline import (
    project_adapters,
)
from mlops_sales_forecasting.pipeline.project_adapters import (
    PersistingRossmannDataIngestor,
    PersistingRossmannFeatureBuilder,
)


def test_persisting_ingestor_writes_canonical_data(
    monkeypatch,
) -> None:
    datasets = DatasetCollection(
        datasets={
            "train": pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
            "store": pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
            "simulation_truth": pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
        }
    )
    base_ingest = MagicMock(return_value=datasets)
    persist_simulation = MagicMock()
    persist_validated = MagicMock()

    monkeypatch.setattr(
        project_adapters.RossmannDataIngestor,
        "ingest",
        base_ingest,
    )
    monkeypatch.setattr(
        project_adapters,
        "persist_simulation_source_if_missing",
        persist_simulation,
    )
    monkeypatch.setattr(
        project_adapters,
        "persist_validated_datasets",
        persist_validated,
    )

    result = PersistingRossmannDataIngestor().ingest(
        {
            "paths": {
                "raw_data": "data/raw",
                "validated_data": ("data/validation"),
            },
        }
    )

    assert result is datasets
    persist_simulation.assert_called_once()
    persist_validated.assert_called_once()


def test_persisting_feature_builder_writes_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    train = pd.DataFrame(
        {
            "Store": [
                1,
            ],
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                ]
            ),
            "Sales": [
                100.0,
            ],
            "StateHoliday": [
                "0",
            ],
            "SchoolHoliday": [
                0,
            ],
        }
    )
    simulation_truth = train.copy()
    simulation_truth["Date"] = pd.to_datetime(
        [
            "2026-01-02",
        ]
    )
    simulation_truth["Sales"] = [
        110.0,
    ]
    datasets = DatasetCollection(
        datasets={
            "train": train,
            "store": pd.DataFrame(
                {
                    "Store": [
                        1,
                    ],
                }
            ),
            "simulation_truth": simulation_truth,
        }
    )
    calendar = pd.DataFrame(
        {
            "Store": [
                1,
                1,
            ],
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "days_until_state_holiday": [
                0,
                0,
            ],
            "days_since_state_holiday": [
                0,
                0,
            ],
            "is_day_before_state_holiday": [
                0,
                0,
            ],
            "is_day_after_state_holiday": [
                0,
                0,
            ],
            "days_until_school_holiday_start": [
                0,
                0,
            ],
            "days_since_school_holiday_start": [
                0,
                0,
            ],
            "days_until_school_holiday_end": [
                0,
                0,
            ],
            "days_since_school_holiday_end": [
                0,
                0,
            ],
            "is_school_holiday_start": [
                0,
                0,
            ],
            "is_school_holiday_end": [
                0,
                0,
            ],
        }
    )
    features = train.assign(sales_lag_1=0.0)

    build_calendar = MagicMock(return_value=calendar)
    monkeypatch.setattr(
        project_adapters,
        "build_known_calendar",
        build_calendar,
    )

    build_features = MagicMock(return_value=features)
    monkeypatch.setattr(
        project_adapters.RossmannFeatureBuilder,
        "build_features",
        build_features,
    )

    create_state = MagicMock()
    monkeypatch.setattr(
        project_adapters,
        "create_feature_state",
        create_state,
    )

    result = PersistingRossmannFeatureBuilder().build_features(
        datasets,
        {
            "paths": {
                "features": str(tmp_path / "features"),
                "models": str(tmp_path / "models"),
            },
            "data": {
                "id_columns": [
                    "Store",
                ],
                "time_column": "Date",
            },
        },
    )

    assert result is features
    assert (tmp_path / "features" / "features.parquet").is_file()
    assert (tmp_path / "features" / "known_calendar.parquet").is_file()
    create_state.assert_called_once()
    build_calendar.assert_called_once()

    calendar_source = build_calendar.call_args.args[0]

    assert calendar_source["Date"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]
    assert calendar_source["Sales"].tolist() == [
        100.0,
        110.0,
    ]
    assert build_calendar.call_args.kwargs["entity_column"] == "Store"
    assert build_calendar.call_args.kwargs["date_column"] == "Date"

    build_features.assert_called_once()

    feature_datasets = build_features.call_args.args[0]

    pd.testing.assert_frame_equal(
        feature_datasets.require("known_calendar"),
        calendar,
    )
