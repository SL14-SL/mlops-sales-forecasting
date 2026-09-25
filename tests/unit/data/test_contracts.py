from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from mlops_sales_forecasting.data.contracts import (
    DataIngestor,
    DatasetCollection,
    DatasetSplits,
    DatasetSplitter,
    FeatureBuilder,
)


def build_frame(
    values: list[int] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": values or [1, 2],
        }
    )


def test_dataset_collection_returns_required_dataset() -> None:
    customers = build_frame()

    collection = DatasetCollection(
        datasets={
            "customers": customers,
        }
    )

    assert collection.require("customers") is customers


def test_dataset_collection_supports_multiple_tables() -> None:
    observations = build_frame()
    metadata = pd.DataFrame(
        {
            "category": ["a", "b"],
        }
    )

    collection = DatasetCollection(
        datasets={
            "observations": observations,
            "metadata": metadata,
        },
        metadata={
            "source": "test",
        },
    )

    assert collection.require(
        "observations"
    ) is observations
    assert collection.require("metadata") is metadata
    assert collection.metadata["source"] == "test"


def test_dataset_collection_must_not_be_empty() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        DatasetCollection(datasets={})


@pytest.mark.parametrize(
    "name",
    ["", 7, None],
)
def test_dataset_names_must_be_non_empty_strings(
    name: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Dataset names",
    ):
        DatasetCollection(
            datasets={
                name: build_frame(),
            }
        )  # type: ignore[dict-item]


def test_collection_values_must_be_dataframes() -> None:
    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        DatasetCollection(
            datasets={
                "customers": [1, 2, 3],
            }
        )  # type: ignore[dict-item]


def test_require_rejects_missing_dataset() -> None:
    collection = DatasetCollection(
        datasets={
            "customers": build_frame(),
        }
    )

    with pytest.raises(
        KeyError,
        match="Required dataset is missing: labels",
    ):
        collection.require("labels")


def test_dataset_splits_accept_optional_test_data() -> None:
    train = build_frame([1, 2, 3])
    validation = build_frame([4])
    test = build_frame([5])

    splits = DatasetSplits(
        train=train,
        validation=validation,
        test=test,
    )

    assert splits.train is train
    assert splits.validation is validation
    assert splits.test is test


def test_dataset_splits_allow_missing_test_data() -> None:
    splits = DatasetSplits(
        train=build_frame([1, 2]),
        validation=build_frame([3]),
    )

    assert splits.test is None


@pytest.mark.parametrize(
    "split_name",
    ["train", "validation"],
)
def test_required_splits_must_not_be_empty(
    split_name: str,
) -> None:
    values: dict[str, pd.DataFrame] = {
        "train": build_frame([1]),
        "validation": build_frame([2]),
    }
    values[split_name] = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match=(
            f"Dataset split '{split_name}' "
            "must not be empty"
        ),
    ):
        DatasetSplits(**values)


def test_test_split_must_not_be_empty_when_present() -> None:
    with pytest.raises(
        ValueError,
        match="Dataset split 'test' must not be empty",
    ):
        DatasetSplits(
            train=build_frame([1]),
            validation=build_frame([2]),
            test=pd.DataFrame(),
        )


@pytest.mark.parametrize(
    "split_name",
    ["train", "validation", "test"],
)
def test_splits_reject_non_dataframe_values(
    split_name: str,
) -> None:
    values: dict[str, object] = {
        "train": build_frame([1]),
        "validation": build_frame([2]),
        "test": build_frame([3]),
    }
    values[split_name] = [1, 2, 3]

    with pytest.raises(
        TypeError,
        match=(
            f"Dataset split '{split_name}' "
            "must be a pandas DataFrame"
        ),
    ):
        DatasetSplits(
            **values  # type: ignore[arg-type]
        )


class ExampleIngestor:
    def ingest(
        self,
        config: Mapping[str, Any],
    ) -> DatasetCollection:
        return DatasetCollection(
            datasets={
                "observations": build_frame(),
            },
            metadata={
                "environment": config["environment"],
            },
        )


class ExampleFeatureBuilder:
    def build_features(
        self,
        datasets: DatasetCollection,
        config: Mapping[str, Any],
    ) -> pd.DataFrame:
        features = datasets.require(
            "observations"
        ).copy()
        features["enabled"] = bool(
            config["enabled"]
        )
        return features


class ExampleSplitter:
    def split(
        self,
        features: pd.DataFrame,
        config: Mapping[str, Any],
    ) -> DatasetSplits:
        split_index = int(config["split_index"])

        return DatasetSplits(
            train=features.iloc[:split_index],
            validation=features.iloc[split_index:],
        )


def test_structural_pipeline_protocols() -> None:
    assert isinstance(
        ExampleIngestor(),
        DataIngestor,
    )
    assert isinstance(
        ExampleFeatureBuilder(),
        FeatureBuilder,
    )
    assert isinstance(
        ExampleSplitter(),
        DatasetSplitter,
    )


def test_protocol_implementations_form_pipeline() -> None:
    collection = ExampleIngestor().ingest(
        {
            "environment": "test",
        }
    )
    features = ExampleFeatureBuilder().build_features(
        collection,
        {
            "enabled": True,
        },
    )
    splits = ExampleSplitter().split(
        features,
        {
            "split_index": 1,
        },
    )

    assert collection.metadata == {
        "environment": "test",
    }
    assert features["enabled"].tolist() == [
        True,
        True,
    ]
    assert len(splits.train) == 1
    assert len(splits.validation) == 1