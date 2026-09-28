import pandas as pd
import pytest

from mlops_sales_forecasting.simulation.ground_truth import (
    DriftScenario,
)
from mlops_sales_forecasting.simulation.pool import (
    build_simulated_daily_batch,
    count_simulation_days,
    normalize_simulation_pool,
)


def build_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [
                2,
                1,
                2,
                1,
            ],
            "Date": [
                "2026-01-02",
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
            ],
            "Sales": [
                200,
                100,
                120,
                180,
            ],
            "Promo": [
                1,
                0,
                1,
                0,
            ],
        }
    )


def test_normalize_simulation_pool() -> None:
    result = normalize_simulation_pool(build_pool())

    assert result[
        [
            "Date",
            "Store",
        ]
    ].values.tolist() == [
        [
            pd.Timestamp("2026-01-01"),
            1,
        ],
        [
            pd.Timestamp("2026-01-01"),
            2,
        ],
        [
            pd.Timestamp("2026-01-02"),
            1,
        ],
        [
            pd.Timestamp("2026-01-02"),
            2,
        ],
    ]


def test_count_simulation_days() -> None:
    assert count_simulation_days(build_pool()) == 2


def test_build_first_daily_batch() -> None:
    result = build_simulated_daily_batch(
        build_pool(),
        day=1,
        scenario=DriftScenario(),
    )

    assert result.day == 1
    assert result.date == pd.Timestamp("2026-01-01")
    assert result.data["Store"].tolist() == [
        1,
        2,
    ]
    assert result.data["Sales"].tolist() == [
        100,
        120,
    ]


def test_daily_batch_applies_drift() -> None:
    result = build_simulated_daily_batch(
        build_pool(),
        day=2,
        scenario=DriftScenario(
            name=("gradual_promo_shift"),
            drift_start_day=2,
            drift_duration_days=1,
            maximum_base_uplift=0.10,
            maximum_promo_uplift=-0.25,
        ),
    )

    assert result.drift.progress == 1.0
    assert result.data["Sales"].tolist() == [
        198,
        150,
    ]


def test_pool_is_not_modified() -> None:
    pool = build_pool()
    original = pool.copy()

    build_simulated_daily_batch(
        pool,
        day=1,
        scenario=DriftScenario(),
    )

    pd.testing.assert_frame_equal(
        pool,
        original,
    )


def test_day_outside_pool_is_rejected() -> None:
    with pytest.raises(
        IndexError,
        match="exceeds available pool",
    ):
        build_simulated_daily_batch(
            build_pool(),
            day=3,
            scenario=DriftScenario(),
        )


@pytest.mark.parametrize(
    "pool",
    [
        pd.DataFrame(),
        pd.DataFrame(
            {
                "Store": [
                    1,
                ],
                "Date": [
                    "invalid",
                ],
                "Sales": [
                    100,
                ],
                "Promo": [
                    0,
                ],
            }
        ),
    ],
)
def test_invalid_pool_is_rejected(
    pool: pd.DataFrame,
) -> None:
    with pytest.raises(
        (
            KeyError,
            ValueError,
        )
    ):
        normalize_simulation_pool(pool)
