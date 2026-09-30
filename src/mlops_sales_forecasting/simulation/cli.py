from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_sales_forecasting.configs.loader import (
    load_config,
)
from mlops_sales_forecasting.inference.model_manager import (
    ModelManager,
)
from mlops_sales_forecasting.inference.serving import (
    load_active_bundle,
)
from mlops_sales_forecasting.orchestration.lifecycle_adapter import (
    PrefectTrainingLifecycleResult,
    run_prefect_model_lifecycle,
)
from mlops_sales_forecasting.pipeline.project_factory import (
    build_project_training_pipeline,
)

from .baseline import (
    restore_simulation_baseline,
    restore_simulation_champion_alias,
    simulation_baseline_exists,
    simulation_baseline_root,
    snapshot_simulation_baseline,
)
from .ground_truth import DriftScenario
from .reporting import (
    summarize_simulation_comparison,
)
from .runner import run_lifecycle_simulation
from .workspace import (
    SimulationWorkspace,
    prepare_simulation_workspace,
)

_ACTIVE_POINTER_FILENAME = "active_serving_release.json"


def _require_mapping(
    mapping: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = mapping.get(name)

    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a valid '{name}' section.")

    return value


def _require_string(
    mapping: Mapping[str, Any],
    name: str,
) -> str:
    value = mapping.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Simulation setting '{name}' must be a non-empty string.")

    return value


def _positive_integer(
    mapping: Mapping[str, Any],
    name: str,
) -> int:
    value = mapping.get(name)

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Simulation setting '{name}' must be a positive integer.")

    return value


def load_simulation_pool(
    source_path: str | Path,
) -> pd.DataFrame:
    """Load the immutable Rossmann simulation source."""
    path = Path(source_path)

    if not path.is_file():
        raise FileNotFoundError(f"Simulation source not found: {path}")

    return pd.read_csv(
        path,
        parse_dates=[
            "Date",
        ],
        dtype={
            "StateHoliday": str,
        },
    )


def scenario_from_config(
    config: Mapping[str, Any],
) -> DriftScenario:
    """Build the configured controlled-drift scenario."""
    simulation = _require_mapping(
        config,
        "simulation",
    )

    return DriftScenario(
        name=_require_string(
            simulation,
            "scenario",
        ),
        drift_start_day=_positive_integer(
            simulation,
            "drift_start_day",
        ),
        drift_duration_days=_positive_integer(
            simulation,
            "drift_duration_days",
        ),
        maximum_base_uplift=float(
            simulation.get(
                "maximum_base_uplift",
                0.0,
            )
        ),
        maximum_promo_uplift=float(
            simulation.get(
                "maximum_promo_uplift",
                0.0,
            )
        ),
    )


def build_simulation_model_manager(
    *,
    workspace: SimulationWorkspace,
) -> ModelManager:
    """Load serving bundles only from the simulation workspace."""

    def bundle_loader():
        return load_active_bundle(workspace.config)

    return ModelManager(bundle_loader)


def bootstrap_simulation_release(
    workspace: SimulationWorkspace,
) -> PrefectTrainingLifecycleResult:
    """Train and publish the initial isolated simulation champion."""
    pipeline = build_project_training_pipeline(workspace.config)

    result = run_prefect_model_lifecycle(
        pipeline=pipeline,
        mlflow_run_name="simulation-initial-champion",
        mlflow_tags={
            "lifecycle": "simulation",
            "simulation_role": "initial_champion",
        },
    )

    if result.serving_release is None:
        raise RuntimeError("Initial simulation training did not publish a serving release.")

    return result


def prepare_simulation_baseline(
    *,
    config: Mapping[str, Any],
    workspace: SimulationWorkspace,
    rebuild: bool = False,
) -> str:
    """
    Restore an existing baseline or create one from a fresh bootstrap.

    Returns:
        Either ``"restored"`` or ``"created"`` for operational logging.
    """
    baseline_root = simulation_baseline_root(config)

    if not rebuild and simulation_baseline_exists(baseline_root):
        restore_simulation_baseline(
            workspace,
            baseline_root=baseline_root,
        )
        restore_simulation_champion_alias(
            workspace,
            baseline_root=baseline_root,
        )
        return "restored"

    bootstrap_simulation_release(workspace)
    snapshot_simulation_baseline(
        workspace,
        baseline_root=baseline_root,
    )

    return "created"


def _default_output_path(
    *,
    config: Mapping[str, Any],
    retraining_enabled: bool,
) -> Path:
    simulation = _require_mapping(
        config,
        "simulation",
    )
    output_root = Path(
        _require_string(
            simulation,
            "output_path",
        )
    )
    filename = "with_retraining.csv" if retraining_enabled else "without_retraining.csv"

    return output_root / filename


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run the isolated Rossmann lifecycle simulation.")
    )
    parser.add_argument(
        "--config",
        default=None,
        help=("Configuration filename from configs/. Defaults to the active environment."),
    )
    parser.add_argument(
        "--retraining",
        choices=[
            "enabled",
            "disabled",
        ],
        default="disabled",
    )
    parser.add_argument(
        "--maximum-days",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--keep-runtime",
        action="store_true",
        help=("Reuse the existing simulation runtime instead of resetting it."),
    )
    parser.add_argument(
        "--rebuild-baseline",
        action="store_true",
        help=("Retrain and replace the saved initial simulation baseline."),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the configured lifecycle simulation."""
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    simulation = _require_mapping(
        config,
        "simulation",
    )

    configured_days = _positive_integer(
        simulation,
        "maximum_days",
    )
    maximum_days = args.maximum_days if args.maximum_days is not None else configured_days

    if maximum_days < 1:
        raise ValueError("Maximum simulation days must be positive.")

    retraining_enabled = args.retraining == "enabled"
    source_path = _require_string(
        simulation,
        "source_path",
    )
    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(
            config=config,
            retraining_enabled=(retraining_enabled),
        )
    )

    pool = load_simulation_pool(source_path)
    if pool.empty:
        raise ValueError("Simulation source must contain at least one row.")

    scenario = scenario_from_config(config)
    workspace = prepare_simulation_workspace(
        config,
        reset=not args.keep_runtime,
    )
    if args.keep_runtime and args.rebuild_baseline:
        raise ValueError("--keep-runtime and --rebuild-baseline cannot be used together.")

    if args.keep_runtime:
        pointer_path = Path(workspace.config["paths"]["models"]) / "active_serving_release.json"

        if not pointer_path.is_file():
            raise FileNotFoundError(
                "--keep-runtime requires an existing simulation serving release."
            )

        baseline_action = "kept"
    else:
        baseline_action = prepare_simulation_baseline(
            config=config,
            workspace=workspace,
            rebuild=args.rebuild_baseline,
        )

    model_manager = build_simulation_model_manager(
        workspace=workspace,
    )
    initial_bundle = model_manager.load_initial()

    print(
        "Simulation started | "
        f"initial_release={initial_bundle.release_id} | "
        f"baseline={baseline_action} | "
        f"scenario={scenario.name} | "
        f"retraining={args.retraining} | "
        f"maximum_days={maximum_days}"
    )

    result = run_lifecycle_simulation(
        pool=pool,
        scenario=scenario,
        workspace=workspace,
        model_manager=model_manager,
        retraining_enabled=retraining_enabled,
        output_path=output_path,
        maximum_days=maximum_days,
    )

    final_row = result.iloc[-1]

    print(
        "Simulation completed | "
        f"days={len(result)} | "
        f"final_rmse={final_row['rmse']} | "
        f"final_event={final_row['event']} | "
        f"output={output_path}"
    )

    return 0


__all__ = [
    "build_simulation_model_manager",
    "load_simulation_pool",
    "main",
    "scenario_from_config",
    "summarize_simulation_comparison",
    "prepare_simulation_baseline",
]
