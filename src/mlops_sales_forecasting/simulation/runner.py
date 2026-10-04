from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_sales_forecasting.api.schema import (
    PredictionRequest,
)
from mlops_sales_forecasting.data.features.update_state import (
    update_feature_state_from_ground_truth,
)
from mlops_sales_forecasting.inference.model_manager import (
    ModelManager,
)
from mlops_sales_forecasting.inference.prediction_service import (
    PredictionService,
)
from mlops_sales_forecasting.monitoring.monitoring_refresh import (
    refresh_monitoring_signals,
)
from mlops_sales_forecasting.monitoring.retraining_policy import (
    RetrainingAction,
    decide_retraining,
)
from mlops_sales_forecasting.monitoring.retraining_state import (
    build_retraining_state_path,
)
from mlops_sales_forecasting.monitoring.signal_collector import (
    collect_retraining_signals,
)
from mlops_sales_forecasting.monitoring.summary import (
    build_monitoring_summary,
)
from mlops_sales_forecasting.orchestration.retraining_service import (
    run_auto_retraining,
)

from .contracts import SimulationDayResult
from .ground_truth import DriftScenario
from .pool import (
    SimulatedDailyBatch,
    build_simulated_daily_batch,
    count_simulation_days,
)
from .reporting import (
    write_lifecycle_results,
)
from .workspace import SimulationWorkspace

_REQUEST_COLUMNS = [
    "Store",
    "Date",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
]


def _build_prediction_request(
    batch: pd.DataFrame,
) -> PredictionRequest:
    """Build a leakage-safe request from one Ground-Truth batch."""
    missing_columns = [column for column in _REQUEST_COLUMNS if column not in batch.columns]

    if missing_columns:
        raise ValueError(f"Simulation batch is missing inference columns: {missing_columns}.")

    request_frame = batch[_REQUEST_COLUMNS].copy()
    request_frame["Date"] = pd.to_datetime(
        request_frame["Date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    records = request_frame.to_dict(
        orient="records",
    )

    return PredictionRequest(
        inputs=records,
    )


def _initialize_state(
    *,
    model_manager: ModelManager,
    state_path: Path,
) -> None:
    """Seed the isolated state from the initially loaded bundle."""
    if state_path.is_file():
        return

    bundle = model_manager.get_bundle()
    state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    state_path.write_text(
        json.dumps(
            bundle.store_state,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_state(
    state_path: Path,
) -> dict[str, Any]:
    """Load the updated isolated feature state."""
    if not state_path.is_file():
        raise FileNotFoundError(f"Simulation feature state is missing: {state_path}")

    state = json.loads(
        state_path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(state, dict):
        raise ValueError("Simulation feature state must be a JSON object.")

    return state


def _persist_ground_truth_batch(
    batch: SimulatedDailyBatch,
    *,
    workspace: SimulationWorkspace,
) -> Path:
    """Persist Ground Truth only after prediction has completed."""
    output_path = workspace.batch_path / (f"ground_truth_simulation_{batch.day:04d}.csv")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    batch.data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def _performance_values(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    performance = summary.get(
        "performance",
        {},
    )

    if not isinstance(performance, Mapping):
        return {}

    if not bool(
        performance.get(
            "available",
            False,
        )
    ):
        return {}

    return dict(performance)


def _simulation_timestamp(
    value: pd.Timestamp,
) -> pd.Timestamp:
    """Normalize a simulation date to nanosecond UTC."""
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.as_unit("ns")


def _initialize_simulation_retraining_clock(
    *,
    workspace: SimulationWorkspace,
    first_evaluated_at: pd.Timestamp,
) -> None:
    """Persist the logical time of the initial simulation training."""
    paths = workspace.config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError("Simulation config must contain a valid 'paths' section.")

    monitoring_path = paths.get("monitoring")

    if not isinstance(monitoring_path, str) or not monitoring_path.strip():
        raise ValueError("Simulation config must define 'paths.monitoring'.")

    state_path = Path(build_retraining_state_path(monitoring_path))

    if state_path.is_file():
        return

    initial_training_at = _simulation_timestamp(first_evaluated_at).to_pydatetime() - timedelta(
        days=1
    )

    state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "last_retrained_at_utc": (initial_training_at.isoformat()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_simulation_day(
    *,
    batch: SimulatedDailyBatch,
    scenario: DriftScenario,
    workspace: SimulationWorkspace,
    model_manager: ModelManager,
    retraining_enabled: bool,
    prediction_service: PredictionService | None = None,
) -> SimulationDayResult:
    """
    Execute one leak-free forecasting lifecycle day.

    The enforced order is prediction, Ground Truth persistence,
    state update, monitoring refresh and optional retraining.
    """
    evaluated_at = _simulation_timestamp(batch.date)

    _initialize_simulation_retraining_clock(
        workspace=workspace,
        first_evaluated_at=evaluated_at,
    )
    _initialize_state(
        model_manager=model_manager,
        state_path=workspace.state_path,
    )

    service = prediction_service or PredictionService(
        model_manager,
        workspace.config,
    )
    request = _build_prediction_request(batch.data)

    service.predict(
        request,
        request_id=f"simulation-day-{batch.day:04d}",
    )

    ground_truth_path = _persist_ground_truth_batch(
        batch,
        workspace=workspace,
    )

    update_feature_state_from_ground_truth(
        str(ground_truth_path),
        config=workspace.config,
        state_path=str(workspace.state_path),
    )

    updated_state = _load_state(workspace.state_path)
    active_bundle = model_manager.get_bundle()
    model_manager.replace_bundle(
        replace(
            active_bundle,
            store_state=updated_state,
        )
    )

    event: str | None = None
    champion_promoted = False
    candidate_run_id: str | None = None

    if retraining_enabled:
        retraining_result = run_auto_retraining(
            config=workspace.config,
            evaluated_at=evaluated_at,
        )
        candidate_run_id = retraining_result.candidate_run_id
        champion_promoted = retraining_result.champion_promoted

        if retraining_result.status == "retrained":
            event = "retrain"

        if champion_promoted:
            reload_result = model_manager.reload()

            if not reload_result.success:
                raise RuntimeError(
                    f"Promoted simulation release could not be loaded: {reload_result.error}"
                )
    else:
        refresh_monitoring_signals(
            config=workspace.config,
            observed_at=(evaluated_at.to_pydatetime()),
        )
        signals = collect_retraining_signals(
            config=workspace.config,
            evaluated_at=evaluated_at,
        )
        decision = decide_retraining(signals)

        if decision.action is RetrainingAction.TRAIN_CANDIDATE:
            event = "would_retrain"

    summary = build_monitoring_summary(
        config=workspace.config,
        model_manager=model_manager,
    )
    performance = _performance_values(summary)

    return SimulationDayResult(
        day=batch.day,
        cumulative_days=batch.day,
        scenario=scenario.name,
        retraining_enabled=retraining_enabled,
        drift_start_day=scenario.drift_start_day,
        drift_duration_days=(scenario.drift_duration_days),
        maximum_base_uplift=(scenario.maximum_base_uplift),
        maximum_promo_uplift=(scenario.maximum_promo_uplift),
        rmse=performance.get("rmse"),
        mae=performance.get("mae"),
        bias=performance.get("bias"),
        n_samples=performance.get("n_samples"),
        window_start=performance.get("window_start"),
        window_end=performance.get("window_end"),
        event=event,
        champion_promoted=champion_promoted,
        candidate_run_id=candidate_run_id,
        latest_batch_file=ground_truth_path.name,
    )


def run_lifecycle_simulation(
    *,
    pool: pd.DataFrame,
    scenario: DriftScenario,
    workspace: SimulationWorkspace,
    model_manager: ModelManager,
    retraining_enabled: bool,
    output_path: str | Path,
    maximum_days: int | None = None,
) -> pd.DataFrame:
    """Run a complete deterministic lifecycle simulation."""
    available_days = count_simulation_days(pool)

    if maximum_days is None:
        selected_days = available_days
    else:
        if maximum_days < 1:
            raise ValueError("Maximum simulation days must be positive.")

        selected_days = min(
            maximum_days,
            available_days,
        )

    results: list[SimulationDayResult] = []

    for day in range(
        1,
        selected_days + 1,
    ):
        batch = build_simulated_daily_batch(
            pool,
            day=day,
            scenario=scenario,
        )

        result = run_simulation_day(
            batch=batch,
            scenario=scenario,
            workspace=workspace,
            model_manager=model_manager,
            retraining_enabled=retraining_enabled,
        )
        results.append(result)

        write_lifecycle_results(
            results,
            output_path,
        )

    return write_lifecycle_results(
        results,
        output_path,
    )
