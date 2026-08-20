# Forecasting API – Incident Response Runbook

## 1. Purpose

This runbook describes how to diagnose and respond to incidents affecting the
demand forecasting platform in the local Docker Compose environment or the
production-style deployment on Google Cloud Run.

It covers:

- Forecasting API
- Serving releases and model artifacts
- MLflow
- Prometheus and Alertmanager
- Prefect-based training and deployment processes

A model rollback should only be performed when the incident is likely related to a new serving release. Infrastructure, capacity, and network failures are usually not resolved by rolling back a model.

---

## 2. Severity levels

| Severity | Meaning | Examples |
|---|---|---|
| SEV-1 | Predictions are completely unavailable | API unavailable, no active serving bundle |
| SEV-2 | Predictions are partially failing or severely degraded | High 5xx rate, significant latency increase |
| SEV-3 | Degradation without an immediate service outage | Isolated errors, monitoring gaps |

---

## 3. Select the incident environment

Run one of the following setup blocks in the terminal used for diagnosis.
Commands in the remaining sections use these variables.

### 3.1 Local Docker environment

```bash
export INCIDENT_ENV="local"
export API_BASE_URL="http://localhost:8000"
export PROMETHEUS_URL="http://localhost:9090"
```

### 3.2 Google Cloud production demo

Load project variables when they are stored in `.env`:

```bash
set -a
source .env
set +a
```

Set the incident context:

```bash
export INCIDENT_ENV="prod"
export API_BASE_URL="${PRODUCTION_API_URL}"
export PROMETHEUS_URL="http://localhost:9090"
```

The production demo is scraped by the local Prometheus and Grafana stack.
Therefore `PROMETHEUS_URL` remains local unless a separate cloud monitoring
service has been deployed.

Validate the selected context:

```bash
printf 'Environment: %s\nAPI: %s\nPrometheus: %s\n' \
  "$INCIDENT_ENV" \
  "$API_BASE_URL" \
  "$PROMETHEUS_URL"

test -n "$API_BASE_URL"
test -n "$PROMETHEUS_URL"
```

Never copy API keys, access tokens or service-account credentials into an
incident report.

---

## 4. General initial diagnosis

### 4.1 Check service status

#### Local

```bash
docker compose ps
```

Expected status:

- `api`: healthy
- `db`: healthy
- `prometheus`: running
- `alertmanager`: running
- `mlflow`: running
- `prefect`: running

#### Production

```bash
gcloud run services describe forecasting-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(
    metadata.name,
    status.url,
    status.traffic,
    status.conditions,
    template.scaling,
    template.containers.resources
  )'
```

When model loading or registry access is affected, inspect MLflow as well:

```bash
gcloud run services describe mlflow-server \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(
    metadata.name,
    status.url,
    status.traffic,
    status.conditions,
    template.scaling,
    template.containers.resources
  )'
```

### 4.2 Check liveness and readiness

```bash
curl -i ${API_BASE_URL}/livez
curl -i ${API_BASE_URL}/readyz
```

Interpretation:

- `/livez = 200`: The API process is running.
- `/readyz = 200`: A complete serving bundle is active.
- `/livez = 200`, `/readyz = 503`: The API is running in degraded mode.
- Both endpoints are unreachable: Process, container, or network failure.

### 4.3 Record the active release lineage

```bash
curl -s ${API_BASE_URL}/readyz |
jq '{
  release_id,
  model_name,
  model_version,
  model_run_id,
  serving_alias
}'
```

Record these values in the incident log.

### 4.4 Inspect API logs

#### Local

```bash
docker compose logs \
  --since=30m \
  --tail=500 \
  api
```

#### Production

```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="forecasting-api"' \
  --project "$GCP_PROJECT_ID" \
  --freshness=30m \
  --limit=500 \
  --order=asc \
  --format='value(timestamp,severity,textPayload,jsonPayload.message)'
```

Filter production errors and failed requests:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="forecasting-api"
   AND (severity>=ERROR OR httpRequest.status>=500)' \
  --project "$GCP_PROJECT_ID" \
  --freshness=30m \
  --limit=200 \
  --order=asc \
  --format='value(timestamp,severity,httpRequest.status,textPayload,jsonPayload.message)'
```

### 4.5 Inspect active alerts

```bash
curl -s \
  ${PROMETHEUS_URL}/api/v1/alerts |
jq '.data.alerts[] | {
  alert: .labels.alertname,
  environment: .labels.environment,
  instance: .labels.instance,
  state: .state,
  severity: .labels.severity,
  value: .value,
  active_at: .activeAt
}'
```

When Prometheus scrapes multiple environments, confirm that the alert's
`environment` and `instance` labels match the incident.

---

## 5. ForecastingAPIUnavailable

### Meaning

Prometheus cannot successfully scrape the API, or the API is unreachable.

Potential impact:

- Predictions are unavailable.
- Health checks fail.
- Downstream applications receive network or server errors.

### Diagnosis

Common check:

```bash
curl -i "${API_BASE_URL}/livez"
```

#### Local

```bash
docker compose ps api
docker compose logs --since=30m --tail=500 api
```

Check dependencies:

```bash
docker compose ps db mlflow prefect
```

Test the API from inside its container:

```bash
docker compose exec -T api \
  curl -fsS http://localhost:8080/livez
```

#### Production

List revisions and inspect current traffic routing:

```bash
gcloud run revisions list \
  --service forecasting-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"

gcloud run services describe forecasting-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format=json |
jq '.status.traffic'
```

Use the production logging commands from section 4.4 to identify startup-probe,
memory, import, IAM or artifact-loading failures.

### Immediate actions

#### Local

If the API container is not running:

```bash
docker compose up -d api
```

If the process appears to be stuck:

```bash
docker compose restart api
```

Wait for liveness:

```bash
until curl -fsS \
  "${API_BASE_URL}/livez" >/dev/null
do
  echo "Waiting for API..."
  sleep 2
done
```

#### Production

Cloud Run has no direct restart operation. Determine whether the failure is
caused by the application revision, runtime configuration or platform capacity.

Redeploy a known-good image through the normal GitHub Actions workflow. If the
latest application revision is faulty and immediate recovery is required,
traffic can be returned to a recorded known-good revision:

```bash
gcloud run services update-traffic forecasting-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --to-revisions '<known-good-revision>=100'
```

Record the previous traffic configuration before changing it. Reconcile any
emergency change with CI/CD and Terraform afterward.

### Rollback criteria

A model rollback is generally not appropriate when `/livez` fails. This usually indicates an infrastructure, startup, or configuration problem.

Consider a rollback only if:

- The API outage started immediately after a serving release.
- The logs identify loading the new model as the cause.
- The previous release is known to be loadable.

### Recovery verification

```bash
curl -fsS "${API_BASE_URL}/livez"
curl -fsS "${API_BASE_URL}/readyz" | jq .
```

Use `make predict-test` locally or `make verify-prod` for production.

### Escalate when

- The API still fails to start after a restart.
- Model artifacts or MLflow are unavailable.
- Cloud Run repeatedly terminates the instance for memory exhaustion.
- IAM, networking or regional platform issues are suspected.
- Persistent volumes, GCS objects or registry metadata may be damaged.

---

## 6. ForecastingServingBundleNotReady

### Meaning

The API process is running, but no complete and validated serving bundle is active.

Potential causes:

- The model artifact is missing.
- The release manifest is invalid.
- An artifact checksum does not match.
- Store metadata, forecasting state, or calendar data is missing.
- The MLflow model version is unavailable.

### Diagnosis

```bash
curl -fsS "${API_BASE_URL}/readyz" | jq .
```

Inspect logs using the environment-appropriate command from section 4.4.

List serving releases through the API:

```bash
curl -fsS \
  -H "X-API-Key: ${API_KEY}" \
  "${API_BASE_URL}/admin/serving-releases" |
jq .
```

#### Local

Inspect the active release pointer:

```bash
jq . models/active_serving_release.json
```

List serving release files:

```bash
find models/serving_releases \
  -maxdepth 2 \
  -type f |
sort
```

Inspect the active manifest:

```bash
RELEASE_ID=$(
  jq -r '.release_id' \
  models/active_serving_release.json
)

jq . \
  "models/serving_releases/${RELEASE_ID}/serving_manifest.json"
```

#### Production

List production release objects:

```bash
gcloud storage ls \
  "gs://${GCP_BUCKET_NAME}/models/serving_releases/"
```

After selecting the release ID reported by `/readyz` or the release-listing
endpoint, inspect its manifest:

```bash
gcloud storage cat \
  "gs://${GCP_BUCKET_NAME}/models/serving_releases/${RELEASE_ID}/serving_manifest.json" |
jq .
```

Inspect both forecasting-API and MLflow production logs when the exact model
version cannot be loaded.

### Immediate actions

If all artifacts are present and consistent, atomically reload the serving
state in either environment:

```bash
curl -fsS \
  -X POST \
  -H "X-API-Key: ${API_KEY}" \
  "${API_BASE_URL}/admin/reload-serving-state" |
jq .
```

Do not edit a published release or its manifest in place.

### Rollback criteria

Perform a rollback if:

- The current release cannot be loaded completely.
- Artifact checksum validation fails.
- The referenced model version is unavailable.
- The previous release was successfully verified.

List available releases:

```bash
curl -fsS \
  -H "X-API-Key: ${API_KEY}" \
  "${API_BASE_URL}/admin/serving-releases" |
jq .
```

Roll back to a known working release using the rollback command defined by the project.

Example:

```bash
make rollback-serving \
  RELEASE_ID=<previous-release-id>
```

Before executing this command, verify the exact variable expected by the Makefile target.

For production, use the authenticated rollback endpoint and its current schema
from `${API_BASE_URL}/docs`. A typical request is:

```bash
curl -fsS \
  -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"release_id":"<previous-release-id>"}' \
  "${API_BASE_URL}/admin/rollback-serving-release" |
jq .
```

### Recovery verification

```bash
curl -fsS "${API_BASE_URL}/readyz" |
jq '{status, release_id, model_version, model_run_id}'
```

Then run `make predict-test` locally or `make verify-prod` for production.

Confirm that `release_id`, `model_version`, and `model_run_id` belong to the restored release.

### Escalate when

- The previous release also cannot be loaded.
- Multiple serving releases have damaged artifacts.
- The MLflow registry and serving manifest report conflicting lineage.
- GCS or MLflow IAM prevents artifact access.

---

## 7. ForecastingPredictionServerErrorRateHigh

### Meaning

More than the configured percentage of `/predict` requests return an HTTP 5xx response.

### Diagnosis

Inspect the alert:

```bash
curl -s ${PROMETHEUS_URL}/api/v1/alerts |
jq '.data.alerts[] |
  select(
    .labels.alertname ==
    "ForecastingPredictionServerErrorRateHigh"
  )'
```

Record the active release lineage:

```bash
curl -s ${API_BASE_URL}/readyz |
jq '{
  release_id,
  model_version,
  model_run_id
}'
```

Search local API logs for prediction failures:

```bash
docker compose logs \
  --since=30m \
  api |
grep -E \
  "Prediction failed|ERROR|Traceback"
```

For production, use the Cloud Run logging commands from section 4.4 and filter
for prediction exceptions, HTTP 5xx responses and the affected revision.

Inspect response status metrics:

```bash
curl -sG \
  --data-urlencode \
  'query=sum by (status_code) (
    increase(
      api_response_status_total{
        path="/predict"
      }[10m]
    )
  )' \
  "${PROMETHEUS_URL}/api/v1/query" |
jq .
```

Run an environment-appropriate semantic prediction test:

```bash
if [ "$INCIDENT_ENV" = "prod" ]; then
  make verify-prod
else
  make predict-test
fi
```

### Immediate actions

- Identify failing request payloads.
- Determine whether only specific stores, dates, or data segments are affected.
- Inspect model, feature, calendar, and state artifacts.
- Compare the start of the errors with the latest release time.

### Rollback criteria

A rollback is appropriate if:

- Errors started immediately after a new release.
- The semantic prediction probe fails.
- The previous model version successfully handles the same request.
- Model or release lineage is inconsistent.

Do not roll back solely because of:

- Invalid client payloads.
- Infrastructure failures.
- General resource exhaustion without a release correlation.

### Recovery verification

```bash
if [ "$INCIDENT_ENV" = "prod" ]; then
  make verify-prod
else
  make predict-test
fi
```

Check the current server-error rate:

```bash
curl -sG \
  --data-urlencode \
  'query=100 *
    (
      sum(
        rate(
          api_response_status_total{
            path="/predict",
            status_code=~"5.."
          }[5m]
        )
      )
      or vector(0)
    )
    /
    clamp_min(
      sum(
        rate(
          api_request_count_total{
            path="/predict"
          }[5m]
        )
      ),
      0.000001
    )' \
  "${PROMETHEUS_URL}/api/v1/query" |
jq .
```

### Escalate when

- The 5xx rate remains elevated after a rollback.
- Failures cannot be isolated to specific inputs.
- Training and serving features may be calculated differently.
- Multiple releases produce the same failures.
- Failures correlate with a Cloud Run platform, IAM or regional issue.

---

## 8. ForecastingPredictionLatencyHigh

### Meaning

The p95 latency of the `/predict` endpoint exceeds the configured SLO.

### Diagnosis

Query p95 prediction latency:

```bash
curl -sG \
  --data-urlencode \
  'query=histogram_quantile(
    0.95,
    sum by (le) (
      rate(
        api_request_latency_seconds_bucket{
          path="/predict"
        }[5m]
      )
    )
  )' \
  "${PROMETHEUS_URL}/api/v1/query" |
jq .
```

Check the number of observations before interpreting a percentile:

```bash
curl -sG \
  --data-urlencode \
  'query=sum(
    increase(
      api_request_latency_seconds_count{
        path="/predict"
      }[15m]
    )
  )' \
  "${PROMETHEUS_URL}/api/v1/query" |
jq .
```

Check prediction traffic:

```bash
curl -sG \
  --data-urlencode \
  'query=60 *
    sum(
      rate(
        api_request_count_total{
          path="/predict"
        }[5m]
      )
    )' \
  "${PROMETHEUS_URL}/api/v1/query" |
jq .
```

One request containing 100 rows is one latency observation, not 100
observations. Separate large-batch latency from representative single-row
online inference before declaring an incident.

#### Local resource diagnosis

```bash
docker stats --no-stream
```

Search for request timing information:

```bash
docker compose logs \
  --since=30m \
  api |
grep -E \
  "Prediction completed|timing_ms"
```

#### Production resource diagnosis

Inspect Cloud Run resources, scaling and traffic:

```bash
gcloud run services describe forecasting-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(
    template.scaling,
    template.containers.resources,
    status.traffic
  )'
```

Search for request timing, memory failures and timeouts:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="forecasting-api"' \
  --project "$GCP_PROJECT_ID" \
  --freshness=30m \
  --limit=500 \
  --format='value(timestamp,severity,textPayload,jsonPayload.message)' |
grep -E "Prediction completed|timing_ms|Memory limit|deadline"
```

Compare `metadata.timing_ms.total` from a prediction response with Prometheus
end-to-end latency. A large difference suggests network, cold-start or platform
overhead. A similarly high internal value identifies application processing as
the primary cause.

### Immediate actions

- Identify large batches or sudden traffic spikes.
- Check CPU and memory utilization.
- Ensure no MLflow or filesystem access occurs in the prediction request path.
- Determine which internal processing stage causes the delay.
- Distinguish cold-start observations from warm steady-state latency.
- Reduce traffic or increase resources if required.

### Rollback criteria

Roll back only when:

- Latency increased immediately after a model change.
- The new model has measurably slower inference.
- The previous model met the SLO under comparable traffic.

A rollback is usually ineffective for:

- Traffic spikes.
- CPU or memory exhaustion.
- Slow external services.
- Unusually large request batches.
- Sparse histogram observations.

### Recovery verification

```bash
if [ "$INCIDENT_ENV" = "prod" ]; then
  make verify-prod
else
  make predict-test
  docker stats --no-stream
fi
```

Observe the p95 latency for at least one complete alert evaluation window.
For production, use controlled single-row probes rather than large-batch stress
tests when validating the online prediction SLO.

### Escalate when

- p95 latency remains elevated under normal traffic.
- The API repeatedly reaches memory or CPU limits.
- Latency occurs outside the application timing stages.
- Horizontal scaling or architectural changes are required.

---

## 9. Incident closure

An incident may be closed when:

- `/livez` and `/readyz` return successful responses.
- `make predict-test` or `make verify-prod` succeeds for the affected environment.
- The active release lineage has been recorded.
- The affected alert has returned to `inactive`.
- Metrics remain stable for at least one complete alert window.
- Emergency Cloud Run traffic changes have been reconciled with CI/CD and
  Terraform.

Document:

- Environment
- Incident start and end time
- Triggered alert
- Cloud Run revision, where applicable
- Release ID before and after remediation
- Model version and run ID
- Root cause
- Actions performed
- Whether a model release or application revision rollback was performed
- Follow-up tasks and owner

## Related Documentation

- [System architecture](architecture.md)
- [Local development](local-development.md)
- [Google Cloud production demo](production-demo.md)
- [Serving releases and rollback](serving-releases.md)
- [Monitoring, SLOs and alerting](monitoring-and-slos.md)
