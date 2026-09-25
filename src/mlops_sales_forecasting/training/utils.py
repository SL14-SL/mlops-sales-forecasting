from collections.abc import Mapping
from typing import Any


def build_drop_columns(
    config: Mapping[str, Any],
) -> list[str]:
    """Build the feature drop list without duplicates."""
    data_config = config.get(
        "data",
        {},
    )
    feature_config = config.get(
        "features",
        {},
    )

    target_column = data_config.get("target_column")

    if not isinstance(target_column, str) or not target_column:
        raise ValueError("Config must define data.target_column.")

    known_targets = data_config.get(
        "known_targets",
        [],
    )
    configured_drop_columns = feature_config.get(
        "drop_columns",
        [],
    )
    time_column = data_config.get("time_column")

    drop_columns = [
        *configured_drop_columns,
        *known_targets,
        target_column,
    ]

    if isinstance(time_column, str) and time_column:
        drop_columns.append(time_column)

    return list(dict.fromkeys(str(column) for column in drop_columns))
