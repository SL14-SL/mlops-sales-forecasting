from pathlib import Path

import pytest

from mlops_sales_forecasting.simulation.workspace import (
    build_simulation_config,
    prepare_simulation_workspace,
)


def build_config(
    tmp_path: Path,
) -> dict:
    source_raw = tmp_path / "source" / "raw"
    source_raw.mkdir(
        parents=True,
    )
    (source_raw / "train.csv").write_text(
        (
            "Store,Date,Sales,StateHoliday\n"
            "1,2026-01-01,100,0\n"
            "1,2026-01-02,110,0\n"
            "1,2026-01-03,120,0\n"
        ),
        encoding="utf-8",
    )
    (source_raw / "store.csv").write_text(
        "Store,StoreType\n1,a\n",
        encoding="utf-8",
    )
    (source_raw / "test.csv").write_text(
        "Store,Date\n1,2026-01-02\n",
        encoding="utf-8",
    )

    return {
        "paths": {
            "raw_data": str(source_raw),
            "models": "artifacts/models",
            "monitoring": "data/monitoring",
            "predictions": "data/predictions",
        },
        "simulation": {
            "runtime_path": str(tmp_path / "simulation" / "runtime"),
        },
        "tracking": {
            "experiment_name": "forecasting-dev",
            "model_name": "forecasting-model-dev",
        },
    }


def test_build_simulation_config_isolates_mutable_paths(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)
    runtime_root = tmp_path / "simulation" / "runtime"

    result = build_simulation_config(
        config,
        runtime_root=runtime_root,
    )

    assert result["paths"]["raw_data"] == str(runtime_root / "raw")
    assert result["paths"]["monitoring"] == str(runtime_root / "monitoring")
    assert result["paths"]["predictions"] == str(runtime_root / "predictions")

    assert config["paths"]["raw_data"] != (result["paths"]["raw_data"])
    assert result["tracking"]["experiment_name"] == ("forecasting-dev-simulation")
    assert result["tracking"]["model_name"] == ("forecasting-model-dev-simulation")
    assert config["tracking"]["experiment_name"] == ("forecasting-dev")
    assert config["tracking"]["model_name"] == ("forecasting-model-dev")


def test_prepare_workspace_copies_base_data(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)

    workspace = prepare_simulation_workspace(config)

    source_train = Path(config["paths"]["raw_data"]) / "train.csv"
    copied_train = workspace.raw_path / "train.csv"

    assert copied_train.is_file()
    assert copied_train.read_text(encoding="utf-8") == source_train.read_text(encoding="utf-8")

    assert (workspace.raw_path / "store.csv").is_file()
    assert (workspace.raw_path / "test.csv").is_file()
    assert workspace.batch_path.is_dir()
    assert workspace.state_path.parent.is_dir()

    assert workspace.config["paths"]["raw_data"] == str(workspace.raw_path)
    assert workspace.config["tracking"]["model_name"] == ("forecasting-model-dev-simulation")


def test_prepare_workspace_resets_previous_runtime(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)
    workspace = prepare_simulation_workspace(
        config,
    )
    obsolete = workspace.runtime_root / "obsolete.txt"
    obsolete.write_text(
        "old simulation state",
        encoding="utf-8",
    )

    refreshed = prepare_simulation_workspace(
        config,
        reset=True,
    )

    assert not obsolete.exists()
    assert (refreshed.raw_path / "train.csv").is_file()


def test_prepare_workspace_requires_base_training_data(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)
    source_raw = Path(config["paths"]["raw_data"])
    (source_raw / "train.csv").unlink()

    with pytest.raises(
        FileNotFoundError,
        match="train.csv",
    ):
        prepare_simulation_workspace(
            config,
        )


def test_workspace_rejects_remote_runtime_path(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)
    config["simulation"]["runtime_path"] = "gs://example/simulation"

    with pytest.raises(
        ValueError,
        match="local paths",
    ):
        prepare_simulation_workspace(
            config,
        )
