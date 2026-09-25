import pytest

from mlops_sales_forecasting.training.utils import (
    build_drop_columns,
)


def test_build_drop_columns_uses_runtime_config() -> None:
    result = build_drop_columns(
        {
            "data": {
                "target_column": "Sales",
                "known_targets": [
                    "Sales",
                    "Customers",
                ],
                "time_column": "Date",
            },
            "features": {
                "drop_columns": [
                    "Unused",
                    "Customers",
                ],
            },
        }
    )

    assert result == [
        "Unused",
        "Customers",
        "Sales",
        "Date",
    ]


def test_build_drop_columns_requires_target() -> None:
    with pytest.raises(
        ValueError,
        match="data.target_column",
    ):
        build_drop_columns(
            {
                "data": {},
            }
        )
