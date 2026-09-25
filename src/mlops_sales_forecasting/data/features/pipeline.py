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
    create_known_calendar_artifact,
    load_known_calendar,
)
from mlops_sales_forecasting.data.features.create_state import (
    create_feature_state,
)
from mlops_sales_forecasting.storage.filesystem import (
    ensure_dir,
    file_exists,
)
from mlops_sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)


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


def load_feature_inputs(
    *,
    validated_path: str,
    calendar_path: str,
) -> DatasetCollection:
    """Load validated Rossmann inputs and the known calendar."""
    train_path = join_uri(
        validated_path,
        "train.parquet",
    )
    store_path = join_uri(
        validated_path,
        "store.parquet",
    )

    required_paths = [
        train_path,
        store_path,
        calendar_path,
    ]
    missing_paths = [path for path in required_paths if not file_exists(path)]

    if missing_paths:
        raise FileNotFoundError(f"Required feature inputs are missing: {missing_paths}")

    return DatasetCollection(
        datasets={
            "train": pd.read_parquet(train_path),
            "store": pd.read_parquet(store_path),
            "known_calendar": load_known_calendar(calendar_path),
        }
    )


def persist_feature_dataset(
    features: pd.DataFrame,
    *,
    features_path: str,
) -> str:
    """Persist the canonical feature dataset."""
    ensure_dir(features_path)
    output_path = join_uri(
        features_path,
        "features.parquet",
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Feature dataset persisted | path=%s | rows=%s",
        output_path,
        len(features),
    )

    return output_path


def run_feature_pipeline(
    config: Mapping[str, Any],
) -> dict[str, str | int]:
    """Build and persist features, calendar and forecasting state."""
    raw_path = _require_path(
        config,
        "raw_data",
    )
    validated_path = _require_path(
        config,
        "validated_data",
    )
    features_path = _require_path(
        config,
        "features",
    )
    models_path = _require_path(
        config,
        "models",
    )

    calendar_path = join_uri(
        features_path,
        "known_calendar.parquet",
    )
    feature_output_path = join_uri(
        features_path,
        "features.parquet",
    )
    state_path = join_uri(
        models_path,
        "latest_state.json",
    )

    create_known_calendar_artifact(
        source_path=join_uri(
            raw_path,
            "train.csv",
        ),
        output_path=calendar_path,
    )

    datasets = load_feature_inputs(
        validated_path=validated_path,
        calendar_path=calendar_path,
    )

    features = RossmannFeatureBuilder().build_features(
        datasets,
        config,
    )

    persisted_feature_path = persist_feature_dataset(
        features,
        features_path=features_path,
    )

    create_feature_state(
        config=config,
        features_path=persisted_feature_path,
        state_path=state_path,
    )

    logger.info(
        "Feature pipeline completed | rows=%s | features=%s | state=%s",
        len(features),
        feature_output_path,
        state_path,
    )

    return {
        "feature_path": persisted_feature_path,
        "calendar_path": calendar_path,
        "state_path": state_path,
        "feature_rows": len(features),
    }
