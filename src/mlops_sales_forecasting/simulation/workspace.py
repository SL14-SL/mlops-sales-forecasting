from __future__ import annotations

import shutil
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationWorkspace:
    """Isolated filesystem and configuration for one simulation run."""

    runtime_root: Path
    raw_path: Path
    batch_path: Path
    state_path: Path
    config: dict[str, Any]


def _require_mapping(
    config: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = config.get(name)

    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a valid '{name}' section.")

    return value


def _require_local_path(
    mapping: Mapping[str, Any],
    name: str,
) -> Path:
    value = mapping.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config value '{name}' must be a non-empty string.")

    if value.startswith("gs://"):
        raise ValueError("Lifecycle simulation currently requires local paths.")

    return Path(value)


def _validate_runtime_root(
    runtime_root: Path,
) -> Path:
    resolved = runtime_root.resolve()

    if len(resolved.parts) < 3:
        raise ValueError("Simulation runtime path is too broad.")

    if resolved == Path.cwd().resolve():
        raise ValueError("Simulation runtime path must not be the project root.")

    return resolved


def _copy_required_file(
    source: Path,
    destination: Path,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required simulation source file is missing: {source}")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        source,
        destination,
    )


def build_simulation_config(
    config: Mapping[str, Any],
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    """Return a config whose mutable paths use the runtime workspace."""
    isolated = deepcopy(dict(config))
    paths = isolated.get("paths")

    if not isinstance(paths, dict):
        raise ValueError("Config must contain a mutable 'paths' mapping.")

    path_values = {
        "data_root": runtime_root,
        "raw_data": runtime_root / "raw",
        "validated_data": runtime_root / "validation",
        "interim": runtime_root / "interim",
        "processed": runtime_root / "processed",
        "artifacts": runtime_root / "artifacts",
        "models": runtime_root / "models",
        "features": runtime_root / "features",
        "splits": runtime_root / "splits",
        "versioning": runtime_root / "versioning",
        "monitoring": runtime_root / "monitoring",
        "predictions": runtime_root / "predictions",
    }

    paths.update({name: str(path) for name, path in path_values.items()})
    tracking = isolated.get("tracking")

    if not isinstance(tracking, dict):
        raise ValueError("Config must contain a mutable 'tracking' mapping.")

    experiment_name = tracking.get("experiment_name")
    model_name = tracking.get("model_name")

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("Config must define tracking.experiment_name.")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Config must define tracking.model_name.")

    tracking["experiment_name"] = f"{experiment_name}-simulation"
    tracking["model_name"] = f"{model_name}-simulation"

    simulation = isolated.get("simulation")

    if not isinstance(simulation, dict):
        raise ValueError("Config must contain a mutable 'simulation' mapping.")

    trigger_settings = simulation.get(
        "retraining_triggers",
        {},
    )

    if not isinstance(
        trigger_settings,
        Mapping,
    ):
        raise ValueError("Config value 'simulation.retraining_triggers' must be a mapping.")

    monitoring = isolated.get("monitoring")

    if not isinstance(monitoring, dict):
        raise ValueError("Config must contain a mutable 'monitoring' mapping.")

    retraining = monitoring.get("retraining")

    if not isinstance(retraining, dict):
        raise ValueError("Config must contain a mutable 'monitoring.retraining' mapping.")

    retraining["triggers"] = dict(trigger_settings)

    scheduled_interval_hours = simulation.get("scheduled_interval_hours")

    if scheduled_interval_hours is not None:
        if (
            isinstance(
                scheduled_interval_hours,
                bool,
            )
            or not isinstance(
                scheduled_interval_hours,
                int,
            )
            or scheduled_interval_hours < 1
        ):
            raise ValueError(
                "Config value 'simulation.scheduled_interval_hours' must be a positive integer."
            )

        retraining["scheduled_interval_hours"] = scheduled_interval_hours

    return isolated


def prepare_simulation_workspace(
    config: Mapping[str, Any],
    *,
    reset: bool = True,
) -> SimulationWorkspace:
    """Create an isolated, reproducible lifecycle workspace."""
    paths = _require_mapping(
        config,
        "paths",
    )
    simulation = _require_mapping(
        config,
        "simulation",
    )

    source_raw_path = _require_local_path(
        paths,
        "raw_data",
    )
    runtime_root = _validate_runtime_root(
        _require_local_path(
            simulation,
            "runtime_path",
        )
    )

    if reset and runtime_root.exists():
        shutil.rmtree(runtime_root)

    raw_path = runtime_root / "raw"
    batch_path = raw_path / "new_batches"
    state_path = runtime_root / "models" / "latest_state.json"

    for directory in (
        raw_path,
        batch_path,
        runtime_root / "validation",
        runtime_root / "interim",
        runtime_root / "processed",
        runtime_root / "artifacts",
        runtime_root / "models",
        runtime_root / "features",
        runtime_root / "splits",
        runtime_root / "versioning",
        runtime_root / "monitoring",
        runtime_root / "predictions",
        state_path.parent,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    for filename in (
        "train.csv",
        "store.csv",
    ):
        _copy_required_file(
            source_raw_path / filename,
            raw_path / filename,
        )

    optional_test = source_raw_path / "test.csv"

    if optional_test.is_file():
        shutil.copy2(
            optional_test,
            raw_path / "test.csv",
        )

    isolated_config = build_simulation_config(
        config,
        runtime_root=runtime_root,
    )

    return SimulationWorkspace(
        runtime_root=runtime_root,
        raw_path=raw_path,
        batch_path=batch_path,
        state_path=state_path,
        config=isolated_config,
    )
