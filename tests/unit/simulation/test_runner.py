import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_sales_forecasting.simulation import runner
from mlops_sales_forecasting.simulation.ground_truth import (
    DriftApplication,
    DriftScenario,
)
from mlops_sales_forecasting.simulation.pool import (
    SimulatedDailyBatch,
)
from mlops_sales_forecasting.simulation.workspace import (
    SimulationWorkspace,
)


@pytest.fixture(autouse=True)
def mock_default_retraining_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "collect_retraining_signals",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        runner,
        "decide_retraining",
        MagicMock(
            return_value=SimpleNamespace(
                action=(runner.RetrainingAction.SKIP),
            )
        ),
    )


def build_batch() -> SimulatedDailyBatch:
    return SimulatedDailyBatch(
        day=3,
        date=pd.Timestamp("2026-01-03"),
        data=pd.DataFrame(
            {
                "Store": [1],
                "Date": pd.to_datetime(["2026-01-03"]),
                "Sales": [120],
                "Open": [1],
                "Promo": [1],
                "StateHoliday": ["0"],
                "SchoolHoliday": [0],
            }
        ),
        drift=DriftApplication(
            progress=0.0,
            base_multiplier=1.0,
            promo_multiplier=1.0,
        ),
    )


def build_workspace(
    tmp_path: Path,
) -> SimulationWorkspace:
    runtime = tmp_path / "simulation" / "runtime"
    batch_path = runtime / "raw" / "new_batches"
    state_path = runtime / "models" / "latest_state.json"

    batch_path.mkdir(
        parents=True,
    )
    state_path.parent.mkdir(
        parents=True,
    )

    return SimulationWorkspace(
        runtime_root=runtime,
        raw_path=runtime / "raw",
        batch_path=batch_path,
        state_path=state_path,
        config={
            "paths": {
                "raw_data": str(runtime / "raw"),
                "models": str(runtime / "models"),
                "monitoring": str(runtime / "monitoring"),
                "predictions": str(runtime / "predictions"),
            },
        },
    )


def build_manager() -> tuple[MagicMock, MagicMock]:
    bundle = MagicMock()
    bundle.store_state = {
        "1": [
            90.0,
            100.0,
        ],
    }

    manager = MagicMock()
    manager.get_bundle.return_value = bundle

    return manager, bundle


def test_prediction_request_excludes_ground_truth() -> None:
    request = runner._build_prediction_request(build_batch().data)

    record = request.inputs[0]

    assert "Sales" not in record
    assert record["Store"] == 1
    assert record["Date"] == "2026-01-03"


def test_day_executes_steps_without_target_leakage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()
    service = MagicMock()

    def predict(*args, **kwargs):
        events.append("predict")
        return MagicMock()

    def persist(*args, **kwargs):
        events.append("ground_truth")
        path = workspace.batch_path / ("ground_truth_simulation_0003.csv")
        path.write_text(
            "Store,Date,Sales\n1,2026-01-03,120\n",
            encoding="utf-8",
        )
        return path

    def update(*args, **kwargs):
        events.append("state")
        workspace.state_path.write_text(
            '{"1": [100.0, 120.0]}',
            encoding="utf-8",
        )
        return {}

    def refresh(*args, **kwargs):
        events.append("monitoring")
        return MagicMock()

    service.predict.side_effect = predict

    monkeypatch.setattr(
        runner,
        "_persist_ground_truth_batch",
        persist,
    )
    monkeypatch.setattr(
        runner,
        "update_feature_state_from_ground_truth",
        update,
    )
    monkeypatch.setattr(
        runner,
        "refresh_monitoring_signals",
        refresh,
    )
    monkeypatch.setattr(
        runner,
        "replace",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        runner,
        "build_monitoring_summary",
        MagicMock(
            return_value={
                "performance": {
                    "available": False,
                },
            }
        ),
    )

    runner.run_simulation_day(
        batch=build_batch(),
        scenario=DriftScenario(),
        workspace=workspace,
        model_manager=manager,
        retraining_enabled=False,
        prediction_service=service,
    )

    assert events == [
        "predict",
        "ground_truth",
        "state",
        "monitoring",
    ]


def test_day_returns_latest_performance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()
    service = MagicMock()

    monkeypatch.setattr(
        runner,
        "update_feature_state_from_ground_truth",
        lambda *args, **kwargs: workspace.state_path.write_text(
            '{"1": [120.0]}',
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        runner,
        "refresh_monitoring_signals",
        MagicMock(),
    )
    monkeypatch.setattr(
        runner,
        "replace",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        runner,
        "build_monitoring_summary",
        MagicMock(
            return_value={
                "performance": {
                    "available": True,
                    "rmse": 800.0,
                    "mae": 600.0,
                    "bias": -25.0,
                    "n_samples": 1115,
                    "window_start": "2026-01-01",
                    "window_end": "2026-01-03",
                },
            }
        ),
    )

    result = runner.run_simulation_day(
        batch=build_batch(),
        scenario=DriftScenario(),
        workspace=workspace,
        model_manager=manager,
        retraining_enabled=False,
        prediction_service=service,
    )

    assert result.rmse == 800.0
    assert result.mae == 600.0
    assert result.bias == -25.0
    assert result.n_samples == 1115


def test_promoted_retraining_reloads_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()
    manager.reload.return_value = SimpleNamespace(
        success=True,
        error=None,
    )

    monkeypatch.setattr(
        runner,
        "update_feature_state_from_ground_truth",
        lambda *args, **kwargs: workspace.state_path.write_text(
            '{"1": [120.0]}',
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        runner,
        "replace",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        runner,
        "run_auto_retraining",
        MagicMock(
            return_value=SimpleNamespace(
                status="retrained",
                candidate_run_id="run-123",
                champion_promoted=True,
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_monitoring_summary",
        MagicMock(
            return_value={
                "performance": {
                    "available": False,
                },
            }
        ),
    )

    result = runner.run_simulation_day(
        batch=build_batch(),
        scenario=DriftScenario(),
        workspace=workspace,
        model_manager=manager,
        retraining_enabled=True,
        prediction_service=MagicMock(),
    )

    assert result.event == "retrain"
    assert result.candidate_run_id == "run-123"
    assert result.champion_promoted is True
    manager.reload.assert_called_once_with()


def build_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [
                1,
                1,
                1,
            ],
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "Sales": [
                100,
                110,
                120,
            ],
            "Open": [
                1,
                1,
                1,
            ],
            "Promo": [
                0,
                1,
                1,
            ],
            "StateHoliday": [
                "0",
                "0",
                "0",
            ],
            "SchoolHoliday": [
                0,
                0,
                0,
            ],
        }
    )


def test_lifecycle_run_processes_days_in_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()
    processed_days: list[int] = []

    def run_day(
        *,
        batch,
        scenario,
        workspace,
        model_manager,
        retraining_enabled,
    ):
        del (
            workspace,
            model_manager,
            retraining_enabled,
        )
        processed_days.append(batch.day)

        return runner.SimulationDayResult(
            day=batch.day,
            cumulative_days=batch.day,
            scenario=scenario.name,
            retraining_enabled=False,
            drift_start_day=(scenario.drift_start_day),
            drift_duration_days=(scenario.drift_duration_days),
            maximum_base_uplift=(scenario.maximum_base_uplift),
            maximum_promo_uplift=(scenario.maximum_promo_uplift),
            rmse=float(1000 - batch.day),
        )

    monkeypatch.setattr(
        runner,
        "run_simulation_day",
        run_day,
    )

    output_path = tmp_path / "results" / "run.csv"

    result = runner.run_lifecycle_simulation(
        pool=build_pool(),
        scenario=DriftScenario(),
        workspace=workspace,
        model_manager=manager,
        retraining_enabled=False,
        output_path=output_path,
    )

    assert processed_days == [
        1,
        2,
        3,
    ]
    assert result["day"].tolist() == [
        1,
        2,
        3,
    ]
    assert output_path.is_file()


def test_lifecycle_run_honors_maximum_days(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()

    monkeypatch.setattr(
        runner,
        "run_simulation_day",
        lambda **kwargs: runner.SimulationDayResult(
            day=kwargs["batch"].day,
            cumulative_days=(kwargs["batch"].day),
            scenario=kwargs["scenario"].name,
            retraining_enabled=False,
            drift_start_day=(kwargs["scenario"].drift_start_day),
            drift_duration_days=(kwargs["scenario"].drift_duration_days),
            maximum_base_uplift=(kwargs["scenario"].maximum_base_uplift),
            maximum_promo_uplift=(kwargs["scenario"].maximum_promo_uplift),
        ),
    )

    result = runner.run_lifecycle_simulation(
        pool=build_pool(),
        scenario=DriftScenario(),
        workspace=workspace,
        model_manager=manager,
        retraining_enabled=False,
        maximum_days=2,
        output_path=tmp_path / "run.csv",
    )

    assert result["day"].tolist() == [
        1,
        2,
    ]


def test_lifecycle_run_rejects_invalid_maximum_days(
    tmp_path: Path,
) -> None:
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        runner.run_lifecycle_simulation(
            pool=build_pool(),
            scenario=DriftScenario(),
            workspace=workspace,
            model_manager=manager,
            retraining_enabled=False,
            maximum_days=0,
            output_path=tmp_path / "run.csv",
        )


def test_disabled_retraining_records_policy_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()

    monkeypatch.setattr(
        runner,
        "update_feature_state_from_ground_truth",
        lambda *args, **kwargs: workspace.state_path.write_text(
            '{"1": [120.0]}',
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        runner,
        "replace",
        MagicMock(return_value=MagicMock()),
    )
    refresh = MagicMock()
    collect = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(
        runner,
        "refresh_monitoring_signals",
        refresh,
    )
    monkeypatch.setattr(
        runner,
        "collect_retraining_signals",
        collect,
    )
    monkeypatch.setattr(
        runner,
        "decide_retraining",
        MagicMock(
            return_value=SimpleNamespace(
                action=(runner.RetrainingAction.TRAIN_CANDIDATE),
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_monitoring_summary",
        MagicMock(
            return_value={
                "performance": {
                    "available": False,
                },
            }
        ),
    )

    result = runner.run_simulation_day(
        batch=build_batch(),
        scenario=DriftScenario(),
        workspace=workspace,
        model_manager=manager,
        retraining_enabled=False,
        prediction_service=MagicMock(),
    )

    assert result.event == "would_retrain"
    expected_time = pd.Timestamp(build_batch().date).tz_localize("UTC")

    refresh.assert_called_once_with(
        config=workspace.config,
        observed_at=(expected_time.to_pydatetime()),
    )
    collect.assert_called_once_with(
        config=workspace.config,
        evaluated_at=expected_time,
    )


def test_simulation_day_uses_logical_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = build_workspace(tmp_path)
    manager, _ = build_manager()
    retrain = MagicMock(
        return_value=SimpleNamespace(
            status="skipped",
            candidate_run_id=None,
            champion_promoted=False,
        )
    )

    monkeypatch.setattr(
        runner,
        "update_feature_state_from_ground_truth",
        lambda *args, **kwargs: workspace.state_path.write_text(
            '{"1": [120.0]}',
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        runner,
        "replace",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        runner,
        "run_auto_retraining",
        retrain,
    )
    monkeypatch.setattr(
        runner,
        "build_monitoring_summary",
        MagicMock(
            return_value={
                "performance": {
                    "available": False,
                },
            }
        ),
    )

    batch = build_batch()

    runner.run_simulation_day(
        batch=batch,
        scenario=DriftScenario(),
        workspace=workspace,
        model_manager=manager,
        retraining_enabled=True,
        prediction_service=MagicMock(),
    )

    expected_time = pd.Timestamp(batch.date).tz_localize("UTC").as_unit("ns")

    retrain.assert_called_once_with(
        config=workspace.config,
        evaluated_at=expected_time,
    )

    state_path = workspace.runtime_root / "monitoring" / "retraining_state.json"
    state = json.loads(
        state_path.read_text(
            encoding="utf-8",
        )
    )
    expected_initial_training_at = expected_time.to_pydatetime() - timedelta(days=1)

    assert state["last_retrained_at_utc"] == (expected_initial_training_at.isoformat())
