import numpy as np
import pandas as pd
import pytest

from src.training.target_transform import (
    transform_target,
    inverse_transform_target,
)


def test_log1p_transform_and_inverse():
    y = pd.Series([0, 10, 100])

    y_transformed = transform_target(y, "log1p")
    y_restored = inverse_transform_target(y_transformed, "log1p")

    assert np.allclose(y, y_restored)


def test_no_transform():
    y = pd.Series([1, 2, 3])

    y_transformed = transform_target(y, "none")

    assert y.equals(y_transformed)


def test_inverse_no_transform_numpy_array():
    y = np.array([1.0, 2.0, 3.0])

    restored = inverse_transform_target(y, "none")

    assert np.array_equal(y, restored)


def test_unsupported_transform_raises():
    y = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        transform_target(y, "unsupported")