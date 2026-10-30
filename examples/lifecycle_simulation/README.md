# Lifecycle simulation reference results

This directory contains small, versioned reference results from the controlled
Rossmann lifecycle simulation.

The large Rossmann source datasets, generated results and simulation runtime
are not stored in Git.

## Reference scenario

The scenario introduces gradual promotional drift:

- drift starts on simulation day 20;
- drift increases over 14 days;
- non-promotional sales remain unchanged;
- promotional sales are reduced by up to 25 percent.

Two otherwise matching lifecycles are compared:

1. the initial champion remains active without retraining;
2. monitoring signals may trigger challenger training and controlled
   promotion.

Retraining and promotion are separate events. Three challengers are trained
on simulation days 30, 38 and 68. The first two are not promoted.
The third passes the configured promotion checks and replaces the active
champion on day 68.

## Reference files

- `without_retraining.csv`: lifecycle under gradual promotional drift without
  executing retraining;
- `with_retraining.csv`: the same lifecycle with automated retraining enabled;
- `segment_metrics.csv`: final comparison for all open stores, promotional
  stores and non-promotional stores.

The checked-in results cover 95 simulation days. Both variants are evaluated
on the same 6,710 open-store observations from July 25–31, 2015.

In this final seven-day window, overall RMSE decreases from `930.79` to
`850.07`, an improvement of `8.67%`. Promotional-store RMSE decreases from
`965.08` to `873.67` (`9.47%`), while non-promotional-store RMSE decreases
from `741.92` to `724.53` (`2.34%`).

These improvements describe this controlled scenario and final evaluation
window, rather than an average improvement over all 95 simulation days.

## Prerequisites

Local input files are required:

```text
data/raw/train.csv
data/raw/store.csv
data/simulation/simulation_ground_truth.csv
```

Start MLflow and Prefect:

```bash
make mlflow-up
make prefect-up

export APP_ENV=dev
export MLFLOW_TRACKING_URI=http://localhost:5000
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

An existing development champion or serving release is not required.
The simulation trains and publishes its own initial release.

## Reproducing the comparison

`configs/lifecycle-reference.yaml` records the model parameters, monitoring
thresholds and scenario settings used for the reference comparison.

Generate a fresh configuration for each independent comparison. Unique
MLflow names prevent an existing champion from affecting initial promotion:

```bash
uv run python - <<'PY'
from pathlib import Path
from uuid import uuid4

import yaml

config = yaml.safe_load(
    Path("configs/lifecycle-reference.yaml").read_text()
)
run_name = f"lifecycle-{uuid4().hex[:12]}"

config["tracking"]["experiment_name"] = run_name
config["tracking"]["model_name"] = f"{run_name}-model"
config["simulation"]["runtime_path"] = f"data/simulation/{run_name}/runtime"
config["simulation"]["baseline_path"] = f"data/simulation/{run_name}/baseline"

Path("configs/lifecycle-local.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8",
)
print(f"Prepared comparison: {run_name}")
PY
```

Run the static baseline:

```bash
uv run python scripts/run_lifecycle_simulation.py \
  --config lifecycle-local.yaml \
  --retraining disabled \
  --maximum-days 95 \
  --output examples/lifecycle_simulation/generated/without_retraining.csv
```

After it completes, run the managed lifecycle with the same configuration:

```bash
uv run python scripts/run_lifecycle_simulation.py \
  --config lifecycle-local.yaml \
  --retraining enabled \
  --maximum-days 95 \
  --output examples/lifecycle_simulation/generated/with_retraining.csv
```

Do not regenerate the configuration between these two runs.
The first run creates the initial baseline; the second restores that same
baseline. Do not use `--rebuild-baseline` or `--keep-runtime` for this paired
comparison.

Each invocation resets the configured runtime workspace. Mutable batches,
monitoring history, feature state and serving pointers remain isolated from
normal development artifacts.

For a separate smoke test, prepare a fresh local configuration and run
with `--maximum-days 7`. Prepare another fresh configuration before starting
the complete comparison.

Generated lifecycle CSV files and companion final-window evaluation Parquet
files are stored under `generated/` and are not committed.
The versioned reference CSV files are updated explicitly after validation.

The dashboard reads the versioned reference CSV files by default.
Running a simulation does not automatically replace those references.

Exact scores can vary with dependency versions and execution environment.
The checked-in reference results document the observed run.

## Dashboard

Start the Streamlit dashboard:

```bash
make dashboard-up
```

Open `http://localhost:8501` and select **Lifecycle Simulation**.

Purple diamonds mark retraining events. The orange star marks the challenger
that passed evaluation and was promoted to champion.

<p align="center">
  <img
    src="../../docs/images/lifecycle-segment-performance.png"
    alt="RMSE comparison for all open, promotional and non-promotional store segments"
    width="900"
  >
</p>

<p align="center">
  <em>
    Across identical observations in the final seven-day evaluation window,
    the managed lifecycle reduces RMSE by 8.67% across all open stores,
    by 9.47% for promotional stores and by 2.34% for non-promotional stores.
  </em>
</p>

