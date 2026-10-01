from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

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


def _require_config_string(
    mapping: Mapping[str, Any],
    name: str,
    *,
    section_name: str,
) -> str:
    value = mapping.get(name)

    if not isinstance(value, str) or not value.strip() or value.startswith("${"):
        raise ValueError(
            f"Config value '{section_name}.{name}' must be a resolved non-empty string."
        )

    return value


def _baseline_model_identity(
    baseline_root: Path,
) -> tuple[str, str]:
    pointer_path = baseline_root / _ACTIVE_POINTER_PATH

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))

    if not isinstance(pointer, dict):
        raise ValueError("Simulation baseline pointer must contain a JSON object.")

    release_id = pointer.get("release_id")

    if not isinstance(release_id, str) or not release_id.strip():
        raise ValueError("Simulation baseline pointer does not contain a valid release ID.")

    manifest_path = (
        baseline_root / "models" / "serving_releases" / release_id / "serving_manifest.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Simulation baseline serving manifest is missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict):
        raise ValueError("Simulation baseline serving manifest must contain a JSON object.")

    model = manifest.get("model")

    if not isinstance(model, dict):
        raise ValueError(
            "Simulation baseline serving manifest does not contain a valid model reference."
        )

    model_name = model.get("name")
    model_version = model.get("version")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Simulation baseline model name is invalid.")

    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("Simulation baseline model version is invalid.")

    return model_name, model_version


def restore_simulation_champion_alias(
    workspace: SimulationWorkspace,
    *,
    baseline_root: Path,
) -> tuple[str, str]:
    """Restore the MLflow serving alias saved in the baseline."""
    runtime_root = workspace.runtime_root.resolve()
    baseline_root = baseline_root.resolve()

    _validate_separate_roots(
        runtime_root=runtime_root,
        baseline_root=baseline_root,
    )

    model_name, model_version = _baseline_model_identity(baseline_root)

    tracking = workspace.config.get("tracking")
    serving = workspace.config.get("serving")

    if not isinstance(tracking, Mapping):
        raise ValueError("Config must contain a valid 'tracking' section.")

    if not isinstance(serving, Mapping):
        raise ValueError("Config must contain a valid 'serving' section.")

    tracking_uri = _require_config_string(
        tracking,
        "mlflow_tracking_uri",
        section_name="tracking",
    )
    serving_alias = _require_config_string(
        serving,
        "alias",
        section_name="serving",
    )

    client = MlflowClient(
        tracking_uri=tracking_uri,
    )
    client.set_registered_model_alias(
        name=model_name,
        alias=serving_alias,
        version=model_version,
    )

    return model_name, model_version


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
