# Lifecycle simulation reference results

This directory contains small, versioned reference results from the controlled
Rossmann lifecycle simulation.

The large Rossmann source datasets and generated simulation runtime are not
stored in Git.

## Reference files

- `without_retraining.csv`: lifecycle under gradual promotional drift without
  executing retraining;
- `with_retraining.csv`: the same lifecycle with automated retraining enabled;
- `segment_metrics.csv`: final comparison for all open stores, promotional
  stores and non-promotional stores.

The reference scenario starts promotional drift on simulation day 20 and
increases it over 14 days until promotional sales are reduced by 25 percent.

## Reproducing a run

The local source data is expected at:

```text
data/simulation/simulation_ground_truth.csv
```

Run a short smoke test:

```bash
uv run python \
  scripts/run_lifecycle_simulation.py \
  --config dev.yaml \
  --retraining disabled \
  --maximum-days 1 \
  --output examples/lifecycle_simulation/generated/smoke-test.csv
```

Run the complete baseline:

```bash
uv run python \
  scripts/run_lifecycle_simulation.py \
  --config dev.yaml \
  --retraining disabled
```

Run the lifecycle with automated retraining:

```bash
uv run python \
  scripts/run_lifecycle_simulation.py \
  --config dev.yaml \
  --retraining enabled
```

Generated runs are written below `generated/` and are intentionally ignored by
Git. The versioned CSV files in this directory remain stable portfolio
references and are not overwritten automatically.