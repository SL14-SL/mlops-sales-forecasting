from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlops_sales_forecasting.configs.paths import (
    join_uri,
)
from mlops_sales_forecasting.data.contracts import (
    DatasetCollection,
)
from mlops_sales_forecasting.data.features.build_features import (
    RossmannFeatureBuilder,
)
from mlops_sales_forecasting.data.features.calendar import (
    build_known_calendar,
)
from mlops_sales_forecasting.data.features.create_state import (
    create_feature_state,
)
from mlops_sales_forecasting.data.raw.ingest import (
    RossmannDataIngestor,
    persist_simulation_source_if_missing,
    persist_validated_datasets,
)
from mlops_sales_forecasting.storage.filesystem import (
    ensure_dir,
)


def _require_path(
    config: Mapping[str, Any],
    name: str,
) -> str:
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a valid 'paths' section.")

    value = paths.get(name)

    if not isinstance(value, str) or not value:
        raise ValueError(f"Config must contain a non-empty 'paths.{name}' value.")

    return value


class PersistingRossmannDataIngestor:
    """Ingest Rossmann data and persist canonical datasets."""

    def ingest(
        self,
        config: Mapping[str, Any],
    ) -> DatasetCollection:
        datasets = RossmannDataIngestor().ingest(config)

        raw_path = _require_path(
            config,
            "raw_data",
        )
        validated_path = _require_path(
            config,
            "validated_data",
        )

        train = datasets.require("train")
        store = datasets.require("store")
        simulation_truth = datasets.require("simulation_truth")

        persist_simulation_source_if_missing(
            simulation_truth,
            raw_path=raw_path,
        )
        persist_validated_datasets(
            train,
            store,
            validated_path=validated_path,
        )

        return datasets


class PersistingRossmannFeatureBuilder:
    """Build features and persist serving-related artifacts."""

    def build_features(
        self,
        datasets: DatasetCollection,
        config: Mapping[str, Any],
    ) -> pd.DataFrame:
        features_path = _require_path(
            config,
            "features",
        )
        models_path = _require_path(
            config,
            "models",
        )
        data_config = config.get(
            "data",
            {},
        )
        id_columns = data_config.get(
            "id_columns",
            [],
        )
        entity_column = str(id_columns[0]) if id_columns else "Store"
        date_column = str(
            data_config.get(
                "time_column",
                "Date",
            )
        )

        known_calendar = datasets.datasets.get("known_calendar")

        if known_calendar is None:
            known_calendar = build_known_calendar(
                datasets.require("train"),
                entity_column=entity_column,
                date_column=date_column,
            )

        augmented_datasets = DatasetCollection(
            datasets={
                **datasets.datasets,
                "known_calendar": (known_calendar),
            },
            metadata=datasets.metadata,
        )

        features = RossmannFeatureBuilder().build_features(
            augmented_datasets,
            config,
        )

        ensure_dir(features_path)
        ensure_dir(models_path)

        calendar_path = join_uri(
            features_path,
            "known_calendar.parquet",
        )
        feature_path = join_uri(
            features_path,
            "features.parquet",
        )
        state_path = join_uri(
            models_path,
            "latest_state.json",
        )

        known_calendar.to_parquet(
            calendar_path,
            index=False,
        )
        features.to_parquet(
            feature_path,
            index=False,
        )

        create_feature_state(
            config=config,
            features_path=feature_path,
            state_path=state_path,
        )

        return features
