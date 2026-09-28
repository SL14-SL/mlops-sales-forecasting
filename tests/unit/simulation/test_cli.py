from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.simulation import cli
from mlops_sales_forecasting.simulation.workspace import (
    SimulationWorkspace,
)


def build_config(
    tmp_path: Path,
) -> dict:
    return {
        "paths": {
            "models": str(tmp_path / "base-models"),
        },
        "simulation": {
            "source_path": str(tmp_path / "simulation.csv"),
            "runtime_path": str(tmp_path / "runtime"),
            "output_path": str(tmp_path / "results"),
            "scenario": "gradual_promo_shift",
            "drift_start_day": 20,
            "drift_duration_days": 14,
            "maximum_base_uplift": 0.0,
            "maximum_promo_uplift": -0.25,
            "maximum_days": 95,
        },
    }


def build_workspace(
    tmp_path: Path,
) -> SimulationWorkspace:
    runtime = tmp_path / "runtime"

    return SimulationWorkspace(
        runtime_root=runtime,
        raw_path=runtime / "raw",
        batch_path=(runtime / "raw" / "new_batches"),
        state_path=(runtime / "models" / "latest_state.json"),
        config={
            "paths": {
                "models": str(runtime / "models"),
            },
        },
    )


def test_load_simulation_pool(
    tmp_path: Path,
) -> None:
    path = tmp_path / "simulation.csv"
    path.write_text(
        "Store,Date,Sales,Promo,StateHoliday\n1,2026-01-01,100,1,0\n",
        encoding="utf-8",
    )

    result = cli.load_simulation_pool(path)

    assert len(result) == 1
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert (
        result.loc[
            0,
            "StateHoliday",
        ]
        == "0"
    )


def test_load_simulation_pool_requires_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Simulation source not found",
    ):
        cli.load_simulation_pool(tmp_path / "missing.csv")


def test_scenario_from_config(
    tmp_path: Path,
) -> None:
    scenario = cli.scenario_from_config(build_config(tmp_path))

    assert scenario.name == ("gradual_promo_shift")
    assert scenario.drift_start_day == 20
    assert scenario.drift_duration_days == 14
    assert scenario.maximum_promo_uplift == -0.25


def test_manager_initially_uses_base_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = build_config(tmp_path)
    workspace = build_workspace(tmp_path)
    base_bundle = MagicMock()

    file_exists = MagicMock(return_value=False)
    load_bundle = MagicMock(return_value=base_bundle)

    monkeypatch.setattr(
        cli,
        "file_exists",
        file_exists,
    )
    monkeypatch.setattr(
        cli,
        "load_active_bundle",
        load_bundle,
    )
    monkeypatch.setattr(
        cli,
        "ModelManager",
        MagicMock(),
    )

    cli.build_simulation_model_manager(
        base_config=base_config,
        workspace=workspace,
    )

    loader = cli.ModelManager.call_args.args[0]
    result = loader()

    assert result is base_bundle
    load_bundle.assert_called_once_with(base_config)


def test_manager_switches_to_simulation_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = build_config(tmp_path)
    workspace = build_workspace(tmp_path)
    simulation_bundle = MagicMock()

    monkeypatch.setattr(
        cli,
        "file_exists",
        MagicMock(return_value=True),
    )
    load_bundle = MagicMock(return_value=simulation_bundle)
    monkeypatch.setattr(
        cli,
        "load_active_bundle",
        load_bundle,
    )
    monkeypatch.setattr(
        cli,
        "ModelManager",
        MagicMock(),
    )

    cli.build_simulation_model_manager(
        base_config=base_config,
        workspace=workspace,
    )

    loader = cli.ModelManager.call_args.args[0]
    result = loader()

    assert result is simulation_bundle
    load_bundle.assert_called_once_with(workspace.config)


def test_default_output_depends_on_retraining(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)

    without = cli._default_output_path(
        config=config,
        retraining_enabled=False,
    )
    with_run = cli._default_output_path(
        config=config,
        retraining_enabled=True,
    )

    assert without.name == ("without_retraining.csv")
    assert with_run.name == ("with_retraining.csv")
