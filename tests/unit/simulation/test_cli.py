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


def test_bootstrap_publishes_initial_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = build_workspace(tmp_path)
    pipeline = MagicMock()
    lifecycle_result = MagicMock()
    lifecycle_result.serving_release = MagicMock()

    build_pipeline = MagicMock(return_value=pipeline)
    run_lifecycle = MagicMock(return_value=lifecycle_result)

    monkeypatch.setattr(
        cli,
        "build_project_training_pipeline",
        build_pipeline,
    )
    monkeypatch.setattr(
        cli,
        "run_prefect_model_lifecycle",
        run_lifecycle,
    )

    result = cli.bootstrap_simulation_release(workspace)

    assert result is lifecycle_result
    build_pipeline.assert_called_once_with(workspace.config)
    run_lifecycle.assert_called_once_with(
        pipeline=pipeline,
        mlflow_run_name=("simulation-initial-champion"),
        mlflow_tags={
            "lifecycle": "simulation",
            "simulation_role": ("initial_champion"),
        },
    )


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


def test_default_evaluation_output_uses_lifecycle_stem(
    tmp_path: Path,
) -> None:
    lifecycle_path = tmp_path / "results" / "with_retraining.csv"

    result = cli._default_evaluation_output_path(lifecycle_path)

    assert result == (tmp_path / "results" / "with_retraining_evaluation.parquet")


def test_manager_only_uses_simulation_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = build_workspace(tmp_path)
    simulation_bundle = MagicMock()

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
        workspace=workspace,
    )

    loader = cli.ModelManager.call_args.args[0]
    result = loader()

    assert result is simulation_bundle
    load_bundle.assert_called_once_with(workspace.config)


def test_existing_baseline_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_config(tmp_path)
    config["simulation"]["baseline_path"] = str(tmp_path / "baseline")
    workspace = build_workspace(tmp_path)
    baseline_root = Path(config["simulation"]["baseline_path"])

    monkeypatch.setattr(
        cli,
        "simulation_baseline_root",
        MagicMock(return_value=baseline_root),
    )
    monkeypatch.setattr(
        cli,
        "simulation_baseline_exists",
        MagicMock(return_value=True),
    )
    restore = MagicMock()
    monkeypatch.setattr(
        cli,
        "restore_simulation_baseline",
        restore,
    )
    restore_alias = MagicMock()
    monkeypatch.setattr(
        cli,
        "restore_simulation_champion_alias",
        restore_alias,
    )
    bootstrap = MagicMock()
    monkeypatch.setattr(
        cli,
        "bootstrap_simulation_release",
        bootstrap,
    )
    snapshot = MagicMock()
    monkeypatch.setattr(
        cli,
        "snapshot_simulation_baseline",
        snapshot,
    )

    result = cli.prepare_simulation_baseline(
        config=config,
        workspace=workspace,
    )

    assert result == "restored"
    restore.assert_called_once_with(
        workspace,
        baseline_root=baseline_root,
    )
    bootstrap.assert_not_called()
    snapshot.assert_not_called()
    restore_alias.assert_called_once_with(
        workspace,
        baseline_root=(tmp_path / "baseline").resolve(),
    )


def test_rebuild_creates_new_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_config(tmp_path)
    config["simulation"]["baseline_path"] = str(tmp_path / "baseline")
    workspace = build_workspace(tmp_path)
    baseline_root = Path(config["simulation"]["baseline_path"])

    monkeypatch.setattr(
        cli,
        "simulation_baseline_root",
        MagicMock(return_value=baseline_root),
    )
    monkeypatch.setattr(
        cli,
        "simulation_baseline_exists",
        MagicMock(return_value=True),
    )
    restore = MagicMock()
    monkeypatch.setattr(
        cli,
        "restore_simulation_baseline",
        restore,
    )
    bootstrap = MagicMock()
    monkeypatch.setattr(
        cli,
        "bootstrap_simulation_release",
        bootstrap,
    )
    snapshot = MagicMock()
    monkeypatch.setattr(
        cli,
        "snapshot_simulation_baseline",
        snapshot,
    )

    result = cli.prepare_simulation_baseline(
        config=config,
        workspace=workspace,
        rebuild=True,
    )

    assert result == "created"
    restore.assert_not_called()
    bootstrap.assert_called_once_with(workspace)
    snapshot.assert_called_once_with(
        workspace,
        baseline_root=baseline_root,
    )
