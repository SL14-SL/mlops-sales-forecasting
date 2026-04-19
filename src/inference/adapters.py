from __future__ import annotations

from typing import Any

import pandas as pd


def request_to_dataframe(inputs: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert generic API inputs into a pandas DataFrame.
    """
    input_df = pd.DataFrame(inputs)
    if input_df.empty:
        raise ValueError("No input rows provided")
    return input_df


def resolve_forecasting_store_id(validated_df: pd.DataFrame) -> int:
    """
    Resolve the forecasting entity key from validated request data.

    This is a temporary forecasting-specific adapter and should later be
    replaced by a config-driven/domain-specific provider.
    """
    if "Store" not in validated_df.columns:
        raise ValueError(
            "Current forecasting backend requires field 'Store'. "
            "Move this assumption into a domain adapter/provider in the next step."
        )

    return int(validated_df["Store"].iloc[0])


def resolve_open_flags(validated_df: pd.DataFrame) -> list[int] | None:
    """
    Extract optional Open flags for forecasting post-processing.
    """
    if "Open" not in validated_df.columns:
        return None

    return [int(v) for v in validated_df["Open"].tolist()]