from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class DatasetCollection:
    """Named data tables produced by an ingestion step."""

    datasets: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError(
                "Dataset collection must not be empty."
            )

        for name, dataset in self.datasets.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Dataset names must be non-empty strings."
                )

            if not isinstance(dataset, pd.DataFrame):
                raise TypeError(
                    f"Dataset '{name}' must be a pandas DataFrame."
                )

    def require(self, name: str) -> pd.DataFrame:
        """Return a required named dataset."""
        try:
            return self.datasets[name]
        except KeyError as exc:
            raise KeyError(
                f"Required dataset is missing: {name}."
            ) from exc


@dataclass(frozen=True)
class DatasetSplits:
    """Train, validation and optional test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self._validate_dataset(
            name="train",
            dataset=self.train,
            allow_none=False,
        )
        self._validate_dataset(
            name="validation",
            dataset=self.validation,
            allow_none=False,
        )
        self._validate_dataset(
            name="test",
            dataset=self.test,
            allow_none=True,
        )

    @staticmethod
    def _validate_dataset(
        *,
        name: str,
        dataset: pd.DataFrame | None,
        allow_none: bool,
    ) -> None:
        if dataset is None:
            if allow_none:
                return

            raise TypeError(
                f"Dataset split '{name}' must be a pandas DataFrame."
            )

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"Dataset split '{name}' must be a pandas DataFrame."
            )

        if dataset.empty:
            raise ValueError(
                f"Dataset split '{name}' must not be empty."
            )


@runtime_checkable
class DataIngestor(Protocol):
    """Load raw project data from its configured source."""

    def ingest(
        self,
        config: Mapping[str, Any],
    ) -> DatasetCollection:
        """Load and return the project's named raw datasets."""
        ...


@runtime_checkable
class FeatureBuilder(Protocol):
    """Create a model-ready feature table."""

    def build_features(
        self,
        datasets: DatasetCollection,
        config: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Transform ingested datasets into model-ready features."""
        ...


@runtime_checkable
class DatasetSplitter(Protocol):
    """Split a model-ready feature table."""

    def split(
        self,
        features: pd.DataFrame,
        config: Mapping[str, Any],
    ) -> DatasetSplits:
        """Create train, validation and optional test splits."""
        ...