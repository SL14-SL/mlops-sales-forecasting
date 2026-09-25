from typing import Any

import pandas as pd


def request_to_dataframe(
    inputs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert API input records into a non-empty DataFrame."""
    input_frame = pd.DataFrame(inputs)

    if input_frame.empty:
        raise ValueError("No input rows provided.")

    return input_frame


def resolve_forecasting_store_id(
    validated_frame: pd.DataFrame,
) -> int:
    """Return the store identifier from one validated request row."""
    if "Store" not in validated_frame.columns:
        raise ValueError(
            "Forecasting inference requires field 'Store'."
        )

    return int(validated_frame["Store"].iloc[0])


def resolve_open_flags(
    validated_frame: pd.DataFrame,
) -> list[int] | None:
    """Extract optional store-open flags for post-processing."""
    if "Open" not in validated_frame.columns:
        return None

    return [
        int(value)
        for value in validated_frame["Open"].tolist()
    ]