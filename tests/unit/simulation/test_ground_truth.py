import pandas as pd
import pytest

from mlops_sales_forecasting.simulation.ground_truth import (
    DriftScenario,
    apply_drift_scenario,
    calculate_drift_application,
)


def build_batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [
                1,
                2,
            ],
            "Sales": [
                100,
                100,
            ],
            "Promo": [
                0,
                1,
            ],
        }
    )


def test_stable_scenario_preserves_sales() -> None:
    batch = build_batch()

    result, application = apply_drift_scenario(
        batch,
        current_day=50,
        scenario=DriftScenario(name="stable"),
    )

    pd.testing.assert_frame_equal(
        result,
        batch,
    )
    assert application.progress == 0.0
    assert application.base_multiplier == 1.0
    assert application.promo_multiplier == 1.0


def test_gradual_drift_is_inactive_before_start() -> None:
    application = calculate_drift_application(
        current_day=19,
        scenario=DriftScenario(
            name=("gradual_promo_shift"),
            drift_start_day=20,
            drift_duration_days=10,
        ),
    )

    assert application.progress == 0.0
    assert application.base_multiplier == 1.0
    assert application.promo_multiplier == 1.0


def test_gradual_drift_ramps_up() -> None:
    result, application = apply_drift_scenario(
        build_batch(),
        current_day=20,
        scenario=DriftScenario(
            name=("gradual_promo_shift"),
            drift_start_day=20,
            drift_duration_days=10,
            maximum_base_uplift=0.10,
            maximum_promo_uplift=0.30,
        ),
    )

    assert application.progress == 0.1
    assert application.base_multiplier == pytest.approx(1.01)
    assert application.promo_multiplier == pytest.approx(1.03)
    assert result["Sales"].tolist() == [
        101,
        103,
    ]


def test_gradual_drift_reaches_full_strength() -> None:
    result, application = apply_drift_scenario(
        build_batch(),
        current_day=40,
        scenario=DriftScenario(
            name=("gradual_promo_shift"),
            drift_start_day=20,
            drift_duration_days=10,
            maximum_base_uplift=0.10,
            maximum_promo_uplift=-0.25,
        ),
    )

    assert application.progress == 1.0
    assert result["Sales"].tolist() == [
        110,
        75,
    ]


def test_input_frame_is_not_modified() -> None:
    batch = build_batch()
    original = batch.copy()

    apply_drift_scenario(
        batch,
        current_day=20,
        scenario=DriftScenario(
            name="gradual_promo_shift",
            drift_start_day=20,
        ),
    )

    pd.testing.assert_frame_equal(
        batch,
        original,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "name": "unknown",
            },
            "Unsupported drift scenario",
        ),
        (
            {
                "drift_start_day": 0,
            },
            "start day",
        ),
        (
            {
                "drift_duration_days": 0,
            },
            "duration",
        ),
        (
            {
                "maximum_promo_uplift": -1.0,
            },
            "promo uplift",
        ),
    ],
)
def test_invalid_scenario_is_rejected(
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        DriftScenario(**kwargs)


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(
        KeyError,
        match="Promo",
    ):
        apply_drift_scenario(
            pd.DataFrame(
                {
                    "Sales": [
                        100,
                    ],
                }
            ),
            current_day=1,
            scenario=DriftScenario(),
        )
