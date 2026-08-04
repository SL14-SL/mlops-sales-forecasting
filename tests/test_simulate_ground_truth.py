import pandas as pd

from scripts.simulate_ground_truth import (
    apply_drift_scenario,
)


def test_stable_scenario_does_not_change_sales():
    batch_df = pd.DataFrame(
        {
            "Promo": [0, 1],
            "Sales": [100, 100],
        }
    )

    result, metadata = apply_drift_scenario(
        batch_df,
        current_day=60,
        scenario="stable",
        drift_start_day=46,
        drift_duration_days=14,
        maximum_base_uplift=0.10,
        maximum_promo_uplift=0.35,
    )

    assert result["Sales"].tolist() == [
        100,
        100,
    ]
    assert metadata["progress"] == 0.0


def test_gradual_promo_shift_reaches_full_strength():
    batch_df = pd.DataFrame(
        {
            "Promo": [0, 1],
            "Sales": [100, 100],
        }
    )

    result, metadata = apply_drift_scenario(
        batch_df,
        current_day=59,
        scenario="gradual_promo_shift",
        drift_start_day=46,
        drift_duration_days=14,
        maximum_base_uplift=0.10,
        maximum_promo_uplift=0.35,
    )

    assert result["Sales"].tolist() == [
        110,
        135,
    ]
    assert metadata["progress"] == 1.0