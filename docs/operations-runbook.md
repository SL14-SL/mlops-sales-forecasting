# Operations runbook

This runbook describes common operational procedures for
**Sales Forecasting MLOps**.

## Scope

The system contains two independently versioned deployment layers:

1. the Cloud Run application revision;
2. the active model serving release.

A Cloud Run rollback does not change the active model. A model rollback
does not change the deployed container revision.

## Required access

Operators may require:

- access to the GitHub repository;
- access to the appropriate GitHub Environment;
- permission to view GitHub Actions;
- Google Cloud Run Viewer or Admin access;
- Google Cloud Logging access;
- access to the artifact bucket;
- access to MLflow when model investigation is required.

Set the common shell variables:

```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_REGION="europe-west1"
export DEPLOYMENT_ENVIRONMENT="dev"
export CLOUD_RUN_SERVICE="mlops-sales-forecasting-${DEPLOYMENT_ENVIRONMENT}-api"
```

## Initial triage

Check the latest deployment workflows:

```bash
gh run list \
  --workflow deploy.yml \
  --limit 10
```

Check rollback workflows:

```bash
gh run list \
  --workflow rollback.yml \
  --limit 10
```

Inspect the Cloud Run service:

```bash
gcloud run services describe \
  "$CLOUD_RUN_SERVICE" \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"
```

List recent revisions:

```bash
gcloud run revisions list \
  --service "$CLOUD_RUN_SERVICE" \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"
```

## Health checks

Obtain the service URI:

```bash
SERVICE_URI="$(
  gcloud run services describe \
    "$CLOUD_RUN_SERVICE" \
    --project "$GCP_PROJECT_ID" \
    --region "$GCP_REGION" \
    --format="value(status.url)"
)"
```

For an IAM-protected service:

```bash
IDENTITY_TOKEN="$(
  gcloud auth print-identity-token
)"
```

Check liveness:

```bash
curl \
  --include \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/livez"
```

A successful liveness response is HTTP `200`.

Check readiness:

```bash
curl \
  --include \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/readyz"
```

Readiness returns HTTP `503` when the application is running but no
valid serving bundle is available.

## Incident: Cloud Run service unavailable

### Symptoms

- `/livez` does not respond;
- Cloud Run reports an unhealthy revision;
- the deployment workflow failed after updating Cloud Run;
- requests return platform-level errors.

### Investigation

Inspect the latest service logs:

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=${CLOUD_RUN_SERVICE}" \
  --project "$GCP_PROJECT_ID" \
  --limit 100 \
  --order desc
```

Inspect revision readiness:

```bash
gcloud run revisions list \
  --service "$CLOUD_RUN_SERVICE" \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"
```

### Recovery

Roll back to the previous revision:

```bash
gh workflow run \
  rollback.yml \
  --field environment="$DEPLOYMENT_ENVIRONMENT"
```

Or select a specific known-good revision:

```bash
gh workflow run \
  rollback.yml \
  --field environment="$DEPLOYMENT_ENVIRONMENT" \
  --field revision="REVISION_NAME"
```

Watch the workflow:

```bash
gh run watch
```

## Incident: application live but not ready

### Symptoms

- `/livez` returns HTTP `200`;
- `/readyz` returns HTTP `503`;
- predictions cannot be served.

### Likely causes

- no serving release has been published;
- the active release pointer is missing;
- the release manifest is invalid;
- a referenced artifact is unavailable;
- checksum verification failed;
- the model loader rejected the model artifact.

### Investigation

Inspect model-loading logs:

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=${CLOUD_RUN_SERVICE} AND severity>=WARNING" \
  --project "$GCP_PROJECT_ID" \
  --limit 100 \
  --order desc
```

Verify that the configured artifact bucket exists:

```bash
gcloud storage buckets describe \
  "gs://${GCP_PROJECT_ID}-mlops-sales-forecasting-${DEPLOYMENT_ENVIRONMENT}-artifacts"
```

List serving-release artifacts:

```bash
gcloud storage ls \
  --recursive \
  "gs://${GCP_PROJECT_ID}-mlops-sales-forecasting-${DEPLOYMENT_ENVIRONMENT}-artifacts/**"
```

### Recovery

- restore the last known-good serving pointer;
- republish the last valid serving release;
- verify artifact checksums;
- restart or redeploy the API after restoring the release.

Do not promote a new model merely to repair an unavailable artifact.
Restore the last verified release first.

## Incident: elevated prediction errors

### Symptoms

- `PredictionErrorRateHigh` is firing;
- prediction responses return HTTP `422`, `503` or `500`;
- structured prediction events show repeated failures.

### Investigation

Distinguish between:

- input validation failures;
- missing active bundles;
- incompatible feature schemas;
- model execution failures;
- application defects.

Inspect recent error logs:

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=${CLOUD_RUN_SERVICE} AND severity>=ERROR" \
  --project "$GCP_PROJECT_ID" \
  --limit 100 \
  --order desc
```

### Recovery

- validation errors: verify the client payload and feature contract;
- readiness errors: restore the serving release;
- application errors: roll back Cloud Run;
- model-specific errors: roll back the model serving release.

## Incident: elevated latency

### Symptoms

- `PredictionLatencyHigh` is firing;
- p95 prediction latency exceeds the configured threshold;
- Cloud Run instances are saturated or frequently cold-starting.

### Investigation

Inspect Cloud Run instance and request logs:

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=${CLOUD_RUN_SERVICE}" \
  --project "$GCP_PROJECT_ID" \
  --limit 100 \
  --order desc
```

Check the current scaling configuration:

```bash
gcloud run services describe \
  "$CLOUD_RUN_SERVICE" \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format yaml
```

### Recovery options

- increase `minimum_instances` to reduce cold starts;
- increase memory or CPU;
- reduce prediction batch size;
- optimize model loading;
- investigate external storage latency;
- roll back a slower application or model revision.

Apply infrastructure changes through Terraform rather than permanently
editing Cloud Run manually.

## Incident: deployment failure

Inspect the failed workflow:

```bash
gh run view \
  --log-failed
```

Common causes include:

- missing GitHub repository variables;
- expired or incorrect Workload Identity configuration;
- missing environment API key;
- insufficient IAM permissions;
- failed container build;
- invalid Terraform configuration;
- an unavailable container image.

A failed plan does not update Cloud Run. A failed apply may require
inspection of Terraform state before retrying.

Do not delete the Terraform state or state bucket to resolve a failed
deployment.

## Incident: Workload Identity authentication failure

Verify the repository variables:

```bash
gh variable list
```

Verify the expected provider and service account:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  output
```

Check that the GitHub repository name and owner match the values used
during bootstrap. Workload Identity is intentionally restricted to one
exact repository.

## Terraform state locking

If Terraform reports an active lock, first verify that no other
deployment or local Terraform command is running.

Do not force-unlock an active deployment.

Only after confirming that the original process no longer exists:

```bash
terraform \
  -chdir=infrastructure/terraform \
  force-unlock \
  LOCK_ID
```

Record the reason whenever a lock is manually removed.

## API key rotation

Update the GitHub Environment secret:

```bash
gh secret set \
  API_KEY \
  --env "$DEPLOYMENT_ENVIRONMENT"
```

Run a reviewed deployment:

```bash
gh workflow run \
  deploy.yml \
  --field environment="$DEPLOYMENT_ENVIRONMENT" \
  --field apply_changes=false
```

After reviewing the plan:

```bash
gh workflow run \
  deploy.yml \
  --field environment="$DEPLOYMENT_ENVIRONMENT" \
  --field apply_changes=true
```

The apply workflow publishes a new Secret Manager version. Cloud Run
continues to reference the latest version.

## Incident: monitoring evidence is stale

### Symptoms

- `/monitoring/summary` reports that performance or feature-drift evidence is
  unavailable;
- the latest performance or drift timestamp is older than expected;
- `performance_rolling.parquet` is missing or unchanged;
- `feature_drift_history.parquet` is missing or unchanged;
- automated retraining repeatedly reports insufficient or absent monitoring
  evidence.

### Investigation

Inspect the persisted monitoring summary:

```bash
curl \
  --fail \
  --silent \
  --show-error \
  "${SERVICE_URI}/monitoring/summary"
```

For an IAM-protected Cloud Run service:

```bash
curl \
  --fail \
  --silent \
  --show-error \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/monitoring/summary"
```

Verify that inference partitions exist:

```bash
gcloud storage ls \
  "gs://${GCS_BUCKET_NAME}/data/predictions/history/**"
```

Verify that delayed Ground-Truth batches exist:

```bash
gcloud storage ls \
  "gs://${GCS_BUCKET_NAME}/data/raw/new_batches/ground_truth_*.csv"
```

Verify the generated monitoring artifacts:

```bash
gcloud storage ls \
  "gs://${GCS_BUCKET_NAME}/data/monitoring/**"
```

Check recent Prefect flow runs:

```bash
uv run prefect flow-run ls \
  --limit 20
```

Inspect API logs for inference persistence failures:

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=${CLOUD_RUN_SERVICE} AND textPayload:\"Could not persist inference monitoring batch\"" \
  --project "$GCP_PROJECT_ID" \
  --limit 100 \
  --order desc
```

Common causes include:

- no successful predictions have been made;
- Ground Truth has not arrived yet;
- prediction and Ground-Truth keys do not overlap;
- the minimum performance sample count has not been reached;
- the configured GCS bucket is unavailable;
- the runtime identity cannot read or write monitoring objects;
- feature-reference data is unavailable;
- a monitoring file has an invalid schema.

### Recovery

1. restore access to the configured storage bucket;
2. verify that new predictions create partitioned Parquet files;
3. upload or restore valid Ground-Truth batches;
4. correct invalid `Store`, `Date` or `Sales` values;
5. run the scheduled auto-retraining deployment only after confirming that a
   real training run is acceptable;
6. verify that the monitoring timestamps advance;
7. verify `/monitoring/summary` again.

Do not fabricate Ground Truth merely to clear monitoring warnings. Missing
labels are preferable to incorrect performance evidence.

## Incident: automated retraining did not behave as expected

### Symptoms

- the scheduled Prefect deployment did not create a flow run;
- a flow run completed with `blocked`, `skipped` or `duplicate`;
- a Candidate was trained unexpectedly;
- repeated flow runs do not start another Candidate;
- the Prefect work pool remains not ready;
- monitoring evidence indicates degradation but no training starts.

### Investigation

Verify the deployment and its schedule:

```bash
uv run prefect deployment inspect \
  mlops-sales-forecasting-auto-retraining/auto-retraining
```

The expected schedule is:

```text
cron: 0 3 * * *
timezone: Europe/Berlin
active: true
```

List the work pool:

```bash
uv run prefect work-pool ls
```

List recent flow runs:

```bash
uv run prefect flow-run ls \
  --limit 20
```

Inspect the persisted retraining state:

```bash
gcloud storage cat \
  "gs://${GCS_BUCKET_NAME}/data/monitoring/retraining_state.json"
```

Check the latest operational evidence:

```bash
curl \
  --fail \
  --silent \
  --show-error \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/monitoring/summary"
```

A `skipped` result is expected when:

- fewer than the configured minimum number of new rows are available;
- the cooldown is active;
- no scheduled, performance or persistent-drift trigger is active;
- monitoring is still in its bootstrap state.

A `blocked` result is expected when:

- Ground-Truth validation failed;
- the configured row budget was exceeded.

A `duplicate` result is expected when the same deterministic decision ID was
already processed successfully.

### Recovery

If the work pool has no active worker, start one in the environment that is
allowed to run training:

```bash
export PREFECT_API_URL="https://your-prefect-server/api"

uv run prefect worker start \
  --pool local-process-pool
```

If the deployment is missing, register the declarative configuration again:

```bash
uv run prefect deploy \
  --all
```

Do not delete `retraining_state.json` merely to force another run. The state
prevents duplicate training for identical evidence.

Before manually starting auto-retraining, verify:

1. the Ground-Truth batches are valid;
2. the number of new rows is within budget;
3. monitoring evidence is current;
4. no recent successful retraining is still inside the cooldown;
5. starting a real MLflow training lifecycle is intended.

Only then start a manual run:

```bash
uv run prefect deployment run \
  mlops-sales-forecasting-auto-retraining/auto-retraining
```

After completion, verify the Prefect result, MLflow Candidate, promotion
decision, serving release and persisted retraining state.

## Post-incident checks

After recovery:

1. verify `/livez`;
2. verify `/readyz`;
3. execute one valid prediction;
4. verify prediction metrics;
5. verify the active Cloud Run revision;
6. verify the active model serving release;
7. document the incident and corrective action.
8. verify `/monitoring/summary`;
9. verify that inference partitions are being created;
10. verify the latest performance and drift timestamps;
11. verify the Prefect deployment schedule;
12. verify that the work pool has an active worker when scheduled training is
    expected;
13. verify the latest persisted retraining decision.

## Escalation information

Capture at least:

- deployment environment;
- service name;
- Cloud Run revision;
- model release ID;
- model name and version;
- request ID;
- approximate incident start time;
- affected endpoints;
- relevant workflow run;
- relevant log excerpts without secrets or customer data.

## Incident: lifecycle notifications are missing

### Symptoms

- training, registration or promotion completes without the expected message;
- lifecycle events appear in logs but not at the configured webhook;
- logs contain `Lifecycle notification delivery failed`.

### Investigation

Confirm that notifications and the webhook are enabled in the active
environment configuration:

```yaml
notifications:
  enabled: true
  webhook:
    enabled: true
```

Verify that `LIFECYCLE_WEBHOOK_URL` is available in the runtime environment.
Do not print the complete value because webhook URLs may contain secret tokens.

Search application or orchestration logs for notification failures:

```text
Lifecycle notification delivery failed
```

Verify that the webhook endpoint:

1. accepts HTTPS POST requests;
2. accepts JSON request bodies;
3. is reachable from the training runtime;
4. responds before the configured timeout;
5. has not expired or been revoked.

### Recovery

- restore network access to the webhook endpoint;
- replace an expired or revoked webhook secret;
- keep `fail_on_error: false` when notification outages must not block model
  training and promotion;
- rerun the lifecycle only when the underlying model operation itself failed.

Notification retries are not performed automatically. The lifecycle event
remains available in the structured application logs.
## Local lifecycle simulation

The lifecycle simulation is a controlled demonstration environment and must
not share mutable paths with normal development or production workloads.

### Preconditions

Verify that MLflow is available:

```bash
curl \
  --fail \
  --silent \
  --show-error \
  http://localhost:5000/health
```

Verify that the active serving pointer exists:

```bash
test \
  -f artifacts/models/active_serving_release.json
```

Verify the simulation source:

```bash
test \
  -f data/simulation/simulation_ground_truth.csv
```

For retraining-enabled runs, start Prefect and configure the local API:

```bash
make prefect-up

export PREFECT_API_URL=http://127.0.0.1:4200/api
```

### Smoke test

```bash
uv run python \
  scripts/run_lifecycle_simulation.py \
  --config dev.yaml \
  --retraining disabled \
  --maximum-days 1 \
  --output examples/lifecycle_simulation/generated/smoke-test.csv
```

### Full comparison

Run the static baseline:

```bash
uv run python \
  scripts/run_lifecycle_simulation.py \
  --config dev.yaml \
  --retraining disabled
```

Run the managed lifecycle:

```bash
uv run python \
  scripts/run_lifecycle_simulation.py \
  --config dev.yaml \
  --retraining enabled
```

Do not pass `--keep-runtime` when producing a new comparison. The default
reset ensures both runs start from a controlled workspace.

### Validate the outputs

```bash
uv run python - <<'PY'
from pathlib import Path

from mlops_sales_forecasting.simulation.reporting import (
    load_lifecycle_results,
    summarize_simulation_comparison,
)

root = Path(
    "examples/lifecycle_simulation/generated"
)

without = load_lifecycle_results(
    root / "without_retraining.csv"
)
with_run = load_lifecycle_results(
    root / "with_retraining.csv"
)

summary = summarize_simulation_comparison(
    without,
    with_run,
)

for name, value in summary.items():
    print(f"{name}: {value}")
PY
```

Expected invariants:

- both runs cover the same scenario and time horizon;
- the enabled run records retraining events;
- promotion occurs only when evaluation approves the challenger;
- the original development serving pointer remains unchanged;
- no target values are used before their simulated arrival.

### Dashboard verification

```bash
make dashboard-up

curl \
  --fail \
  --silent \
  --show-error \
  http://localhost:8501/_stcore/health
```

Open `http://localhost:8501` and select **Lifecycle Simulation**.
