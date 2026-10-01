from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlops_sales_forecasting.simulation.baseline import (
    restore_simulation_baseline,
    restore_simulation_champion_alias,
    simulation_baseline_exists,
    simulation_baseline_root,
    snapshot_simulation_baseline,
)
from mlops_sales_forecasting.simulation.workspace import (
    SimulationWorkspace,
)


def build_workspace(
    tmp_path: Path,
) -> SimulationWorkspace:
    runtime_root = tmp_path / "simulation" / "runtime"
    raw_path = runtime_root / "raw"
    models_path = runtime_root / "models"

    raw_path.mkdir(
        parents=True,
    )
    models_path.mkdir(
        parents=True,
    )

    (models_path / "active_serving_release.json").write_text(
        '{"release_id":"release-test"}',
        encoding="utf-8",
    )
    (models_path / "latest_state.json").write_text(
        '{"1":[100.0]}',
        encoding="utf-8",
    )
    (raw_path / "train.csv").write_text(
        "Store,Date,Sales\n1,2026-01-01,100\n",
        encoding="utf-8",
    )
    release_path = models_path / "serving_releases" / "release-test"
    release_path.mkdir(
        parents=True,
    )
    (release_path / "serving_manifest.json").write_text(
        ('{"model":{"name":"simulation-model","version":"5"}}'),
        encoding="utf-8",
    )

    return SimulationWorkspace(
        runtime_root=runtime_root,
        raw_path=raw_path,
        batch_path=(raw_path / "new_batches"),
        state_path=(models_path / "latest_state.json"),
        config={
            "paths": {
                "models": str(models_path),
            },
            "tracking": {
                "mlflow_tracking_uri": ("http://localhost:5000"),
            },
            "serving": {
                "alias": "champion",
            },
        },
    )


def test_baseline_root_from_config(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "simulation" / "baseline"

    result = simulation_baseline_root(
        {
            "simulation": {
                "baseline_path": str(configured),
            },
        }
    )

    assert result == configured.resolve()


def test_snapshot_creates_complete_baseline(
    tmp_path: Path,
) -> None:
    workspace = build_workspace(tmp_path)
    baseline_root = tmp_path / "simulation" / "baseline"

    snapshot_simulation_baseline(
        workspace,
        baseline_root=baseline_root,
    )

    assert simulation_baseline_exists(baseline_root)
    assert (baseline_root / "raw" / "train.csv").is_file()
    assert (baseline_root / "models" / "active_serving_release.json").is_file()
    assert (baseline_root / "models" / "latest_state.json").is_file()
    assert (
        baseline_root / "models" / "serving_releases" / "release-test" / "serving_manifest.json"
    ).is_file()


def test_restore_replaces_mutated_runtime(
    tmp_path: Path,
) -> None:
    workspace = build_workspace(tmp_path)
    baseline_root = tmp_path / "simulation" / "baseline"

    snapshot_simulation_baseline(
        workspace,
        baseline_root=baseline_root,
    )

    workspace.state_path.write_text(
        '{"1":[999.0]}',
        encoding="utf-8",
    )
    obsolete = workspace.runtime_root / "obsolete.txt"
    obsolete.write_text(
        "mutated",
        encoding="utf-8",
    )

    restore_simulation_baseline(
        workspace,
        baseline_root=baseline_root,
    )

    assert (
        workspace.state_path.read_text(
            encoding="utf-8",
        )
        == '{"1":[100.0]}'
    )
    assert not obsolete.exists()


def test_incomplete_baseline_is_rejected(
    tmp_path: Path,
) -> None:
    workspace = build_workspace(tmp_path)
    baseline_root = tmp_path / "simulation" / "baseline"
    baseline_root.mkdir(
        parents=True,
    )

    with pytest.raises(
        FileNotFoundError,
        match="incomplete",
    ):
        restore_simulation_baseline(
            workspace,
            baseline_root=baseline_root,
        )


def test_nested_baseline_path_is_rejected(
    tmp_path: Path,
) -> None:
    workspace = build_workspace(tmp_path)
    nested_baseline = workspace.runtime_root / "baseline"

    with pytest.raises(
        ValueError,
        match="must not contain each other",
    ):
        snapshot_simulation_baseline(
            workspace,
            baseline_root=nested_baseline,
        )


def test_restore_resets_serving_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = build_workspace(tmp_path)
    baseline_root = tmp_path / "simulation" / "baseline"

    snapshot_simulation_baseline(
        workspace,
        baseline_root=baseline_root,
    )

    client = MagicMock()
    client_factory = MagicMock(return_value=client)

    monkeypatch.setattr(
        "mlops_sales_forecasting.simulation.baseline.MlflowClient",
        client_factory,
    )

    result = restore_simulation_champion_alias(
        workspace,
        baseline_root=baseline_root,
    )

    assert result == (
        "simulation-model",
        "5",
    )
    client_factory.assert_called_once_with(
        tracking_uri="http://localhost:5000",
    )
    client.set_registered_model_alias.assert_called_once_with(
        name="simulation-model",
        alias="champion",
        version="5",
    )
