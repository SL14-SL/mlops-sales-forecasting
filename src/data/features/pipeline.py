import os

import pandas as pd

from src.configs.loader import (
    get_path,
    load_config,
)
from src.configs.paths import join_uri
from src.data.features.build_features import (
    build_features,
)
from src.data.features.calendar import (
    load_known_calendar,
    merge_known_calendar_features,
)
from src.storage.filesystem import file_exists
from src.utils.logger import get_logger


logger = get_logger(__name__)

TRAIN_CFG = load_config(
    "training.yaml"
)

FEATURES_PATH = get_path(
    "features"
)
VALIDATED_PATH = get_path(
    "validated_data"
)


def resolve_entity_column(
    config: dict,
) -> str:
    """
    Resolve the primary forecasting entity column from configuration.

    Raises:
        ValueError: If no entity column is configured.
    """
    data_config = config.get(
        "data",
        {},
    )
    id_columns = data_config.get(
        "id_columns",
        [],
    )

    if not id_columns:
        raise ValueError(
            "Config must define at least one "
            "column in data.id_columns."
        )

    return str(
        id_columns[0]
    )


def load_validated_inputs(
    *,
    validated_path: str = VALIDATED_PATH,
) -> dict[str, pd.DataFrame]:
    """
    Load the validated training observations and store metadata.

    Raises:
        FileNotFoundError: If either required artifact is unavailable.
    """
    train_path = join_uri(
        validated_path,
        "train.parquet",
    )
    store_path = join_uri(
        validated_path,
        "store.parquet",
    )

    if (
        not file_exists(train_path)
        or not file_exists(store_path)
    ):
        raise FileNotFoundError(
            "Validated training or store data is missing | "
            f"train_path={train_path} | "
            f"store_path={store_path}. "
            "Run ingestion before feature generation."
        )

    train = pd.read_parquet(
        train_path
    )
    store = pd.read_parquet(
        store_path
    )

    logger.info(
        "Validated datasets loaded | "
        "train_shape=%s | store_shape=%s",
        train.shape,
        store.shape,
    )

    return {
        "train": train,
        "store": store,
    }


def merge_feature_sources(
    datasets: dict[str, pd.DataFrame],
    *,
    config: dict,
) -> pd.DataFrame:
    """
    Join training observations, store metadata and known calendar features.

    Raises:
        KeyError: If a required dataset is absent.
        ValueError: If configured keys or calendar coverage are invalid.
    """
    entity_column = resolve_entity_column(
        config
    )

    logger.info(
        "Merging validated datasets | "
        "entity_column=%s",
        entity_column,
    )

    merged = datasets["train"].merge(
        datasets["store"],
        on=entity_column,
        how="left",
    )

    calendar = load_known_calendar()

    merged = merge_known_calendar_features(
        merged,
        calendar,
        strict=True,
    )

    logger.info(
        "Known calendar features merged | rows=%s",
        len(merged),
    )

    return merged


def persist_feature_dataset(
    features: pd.DataFrame,
    *,
    features_path: str = FEATURES_PATH,
) -> str:
    """
    Persist the canonical feature dataset.

    Returns:
        The local or GCS path of the written Parquet artifact.
    """
    if not features_path.startswith(
        "gs://"
    ):
        os.makedirs(
            features_path,
            exist_ok=True,
        )

    output_path = join_uri(
        features_path,
        "features.parquet",
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Feature dataset persisted | "
        "path=%s | shape=%s",
        output_path,
        features.shape,
    )

    return output_path


def run_feature_pipeline(
    config: dict | None = None,
) -> None:
    """
    Build and persist the complete training feature dataset.

    The pipeline loads validated observations and store metadata, joins known
    calendar features, applies configured feature engineering and persists the
    resulting canonical feature table.
    """
    effective_config = (
        config
        or TRAIN_CFG
    )

    logger.info(
        "Starting feature pipeline | "
        "validated_path=%s",
        VALIDATED_PATH,
    )

    datasets = load_validated_inputs()

    merged = merge_feature_sources(
        datasets,
        config=effective_config,
    )

    features = build_features(
        merged,
        config=effective_config,
        mode="train",
    )

    persist_feature_dataset(
        features
    )


def build_feature_dataset() -> None:
    """Build features using the default training configuration."""
    run_feature_pipeline(
        config=TRAIN_CFG
    )


if __name__ == "__main__":
    run_feature_pipeline()