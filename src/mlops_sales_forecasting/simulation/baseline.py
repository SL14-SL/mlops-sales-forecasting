from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .workspace import SimulationWorkspace

_ACTIVE_POINTER_PATH = Path(
    "models",
    "active_serving_release.json",
)
_FEATURE_STATE_PATH = Path(
    "models",
    "latest_state.json",
)


def _require_baseline_root(
    config: Mapping[str, Any],
) -> Path:
    simulation = config.get("simulation")

    if not isinstance(simulation, Mapping):
        raise ValueError("Config must contain a valid 'simulation' section.")

    configured_path = simulation.get("baseline_path")

    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("Config must define a non-empty simulation.baseline_path.")

    if configured_path.startswith("gs://"):
        raise ValueError("Lifecycle simulation baselines currently require local paths.")

    resolved = Path(configured_path).resolve()

    if len(resolved.parts) < 3:
        raise ValueError("Simulation baseline path is too broad.")

    if resolved == Path.cwd().resolve():
        raise ValueError("Simulation baseline path must not be the project root.")

    return resolved


def _validate_separate_roots(
    *,
    runtime_root: Path,
    baseline_root: Path,
) -> None:
    runtime = runtime_root.resolve()
    baseline = baseline_root.resolve()

    if runtime == baseline:
        raise ValueError("Simulation runtime and baseline paths must be different.")

    if baseline.is_relative_to(runtime) or runtime.is_relative_to(baseline):
        raise ValueError("Simulation runtime and baseline paths must not contain each other.")


def _missing_baseline_files(
    baseline_root: Path,
) -> list[Path]:
    required = (
        _ACTIVE_POINTER_PATH,
        _FEATURE_STATE_PATH,
    )

    return [
        relative_path for relative_path in required if not (baseline_root / relative_path).is_file()
    ]


def simulation_baseline_root(
    config: Mapping[str, Any],
) -> Path:
    """Return the validated baseline directory."""
    return _require_baseline_root(config)


def simulation_baseline_exists(
    baseline_root: Path,
) -> bool:
    """Return whether a complete baseline is available."""
    return baseline_root.is_dir() and not _missing_baseline_files(baseline_root)


def snapshot_simulation_baseline(
    workspace: SimulationWorkspace,
    *,
    baseline_root: Path,
) -> None:
    """Persist a clean bootstrapped simulation workspace."""
    runtime_root = workspace.runtime_root.resolve()
    baseline_root = baseline_root.resolve()

    _validate_separate_roots(
        runtime_root=runtime_root,
        baseline_root=baseline_root,
    )

    if not runtime_root.is_dir():
        raise FileNotFoundError(f"Simulation runtime does not exist: {runtime_root}")

    missing_runtime_files = _missing_baseline_files(runtime_root)

    if missing_runtime_files:
        missing = ", ".join(str(path) for path in missing_runtime_files)
        raise FileNotFoundError(
            "Simulation runtime cannot be "
            "snapshotted because required files "
            f"are missing: {missing}"
        )

    if baseline_root.exists():
        shutil.rmtree(baseline_root)

    baseline_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copytree(
        runtime_root,
        baseline_root,
    )


def restore_simulation_baseline(
    workspace: SimulationWorkspace,
    *,
    baseline_root: Path,
) -> None:
    """Replace the current runtime with a saved baseline."""
    runtime_root = workspace.runtime_root.resolve()
    baseline_root = baseline_root.resolve()

    _validate_separate_roots(
        runtime_root=runtime_root,
        baseline_root=baseline_root,
    )

    if not simulation_baseline_exists(baseline_root):
        missing = ", ".join(str(path) for path in _missing_baseline_files(baseline_root))
        raise FileNotFoundError(f"Simulation baseline is incomplete. Missing files: {missing}")

    if runtime_root.exists():
        shutil.rmtree(runtime_root)

    runtime_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copytree(
        baseline_root,
        runtime_root,
    )
