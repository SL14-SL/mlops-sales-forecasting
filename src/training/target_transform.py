import numpy as np
import pandas as pd


def transform_target(y: pd.Series, transform: str | None):
    """Apply configured target transformation."""
    if transform in (None, "none"):
        return y
    if transform == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unsupported target transformation: {transform}")


def inverse_transform_target(y, transform: str | None):
    """Revert configured target transformation."""
    if transform in (None, "none"):
        return y
    if transform == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unsupported target transformation: {transform}")