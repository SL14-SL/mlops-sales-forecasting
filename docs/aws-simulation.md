# AWS lifecycle simulation

The `simulate` operation runs the existing isolated Rossmann simulation in the
training Fargate task. It uses the same simulation clock, controlled promo-effect
drift, delayed-ground-truth order, monitoring, cooldown and retraining policy as
the local simulator. The configured full run is 95 days.

The task downloads `train.csv`, `store.csv` and `simulation_ground_truth.csv`
from the configured S3 raw-data directory. Initial AWS training has already
created the simulation source. Missing inputs or insufficient dates fail the run.

The simulation workspace is on the task's ephemeral filesystem. MLflow tracking
and model registration use the configured remote tracking service. Each task
gets unique simulation experiment and model names. In `both` mode, the static
and managed arms restore the same initial baseline; only one bootstrap training
is needed. Simulation promotions and model reloads occur inside this workspace.
The live forecasting API, its registry champion and S3 serving pointer are not
updated. This test does not verify rolling updates of the live API service.

After merging the code, deploy the new training image using `deploy-aws.yml`
with `environment=dev`, `enable_serving=true` and `apply_changes=true`. Wait for
that deployment before starting a simulation, so the existing API stays enabled.

```bash
gh workflow run lifecycle-aws.yml --ref main \
  -f environment=dev \
  -f operation=simulate \
  -f simulation_days=95 \
  -f simulation_mode=both
```

Leave `estimators` empty for the project's full model configuration. `managed`
or `static` selects one arm only. Smaller `simulation_days` values (1–95) are
useful for startup checks but do not establish long-term drift or adaptation.

Each completed day logs its date, RMSE, retraining event and promotion status.
The launcher forwards these messages from CloudWatch into the workflow log.
For live terminal output, use a second terminal:

```bash
aws logs tail /ecs/mlops-sales-forecasting-dev-training \
  --profile mlops-terraform --region eu-north-1 \
  --since 5m --follow --format short
```

Each completed arm uploads its lifecycle CSV and final evaluation Parquet to
`s3://BUCKET/artifacts/lifecycle_simulation/RUN_ID/`. A successful task also
uploads `summary.json`, and prints the exact `results_uri`. In `both` mode,
the summary includes final RMSE for each arm, relative improvement, retraining
events and promotions. The run does not require the managed arm to improve:
performance is an experimental result, not a success condition.

The task deadline is 50 minutes for simulations and 40 minutes for other
lifecycle operations. The workflow allows 60 minutes including setup. The simulation deadline stays below the runner credential session; larger runs
may require a longer session and adjusted deadlines. A new
task starts a new isolated run; task-local baseline and intermediate data are
not resumable. The original S3 inputs remain unchanged. Incomplete arms do not
have guaranteed reports; inspect CloudWatch when a run fails or times out.

This mode has local integration coverage using real XGBoost training, an MLflow
database, baseline restoration and retraining driven by logical time. The S3
transport is substituted in that test. Live AWS execution must be verified by
the workflow above.
