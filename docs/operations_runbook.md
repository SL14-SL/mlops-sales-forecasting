# Forecasting API – Incident Response Runbook

## 1. Purpose

This runbook describes how to diagnose and respond to incidents affecting the demand forecasting platform.

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

## 3. General initial diagnosis

### 3.1 Check service status

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

### 3.2 Check liveness and readiness

```bash
curl -i http://localhost:8000/livez
curl -i http://localhost:8000/readyz
```

Interpretation:

- `/livez = 200`: The API process is running.
- `/readyz = 200`: A complete serving bundle is active.
- `/livez = 200`, `/readyz = 503`: The API is running in degraded mode.
- Both endpoints are unreachable: Process, container, or network failure.

### 3.3 Record the active release lineage

```bash
curl -s http://localhost:8000/readyz |
jq '{
  release_id,
  model_name,
  model_version,
  model_run_id,
  serving_alias
}'
```

Record these values in the incident log.

### 3.4 Inspect API logs

```bash
docker compose logs \
  --since=30m \
  --tail=500 \
  api
```

### 3.5 Inspect active alerts

```bash
curl -s \
  http://localhost:9090/api/v1/alerts |
jq '.data.alerts[] | {
  alert: .labels.alertname,
  state: .state,
  severity: .labels.severity,
  value: .value,
  active_at: .activeAt
}'
```

---

## 4. ForecastingAPIUnavailable

### Meaning

Prometheus cannot successfully scrape the API, or the API is unreachable.

Potential impact:

- Predictions are unavailable.
- Health checks fail.
- Downstream applications receive network or server errors.

### Diagnosis

```bash
docker compose ps api
curl -i http://localhost:8000/livez
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

### Immediate actions

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
  http://localhost:8000/livez >/dev/null
do
  echo "Waiting for API..."
  sleep 2
done
```

### Rollback criteria

A model rollback is generally not appropriate when `/livez` fails. This usually indicates an infrastructure, startup, or configuration problem.

Consider a rollback only if:

- The API outage started immediately after a serving release.
- The logs identify loading the new model as the cause.
- The previous release is known to be loadable.

### Recovery verification

```bash
curl -fsS http://localhost:8000/livez
curl -fsS http://localhost:8000/readyz | jq .
make predict-test
```

### Escalate when

- The API still fails to start after a restart.
- Model artifacts or MLflow are unavailable.
- Persistent volumes may be damaged or data loss is suspected.

---

## 5. ForecastingServingBundleNotReady

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
curl -s http://localhost:8000/readyz | jq .
docker compose logs --since=30m --tail=500 api
```

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

### Immediate actions

If all artifacts are present and consistent, restart the API:

```bash
docker compose restart api
```

Alternatively, reload the serving state through the administrative reload endpoint if the required API key is available.

### Rollback criteria

Perform a rollback if:

- The current release cannot be loaded completely.
- Artifact checksum validation fails.
- The referenced model version is unavailable.
- The previous release was successfully verified.

List available releases:

```bash
make list-serving-releases
```

Roll back to a known working release using the rollback command defined by the project.

Example:

```bash
make rollback-serving \
  RELEASE_ID=<previous-release-id>
```

Before executing this command, verify the exact variable expected by the Makefile target.

### Recovery verification

```bash
curl -fsS http://localhost:8000/readyz | jq .
make predict-test
```

Confirm that `release_id`, `model_version`, and `model_run_id` belong to the restored release.

### Escalate when

- The previous release also cannot be loaded.
- Multiple serving releases have damaged artifacts.
- The MLflow registry and serving manifest report conflicting lineage.

---

## 6. ForecastingPredictionServerErrorRateHigh

### Meaning

More than the configured percentage of `/predict` requests return an HTTP 5xx response.

### Diagnosis

Inspect the alert:

```bash
curl -s http://localhost:9090/api/v1/alerts |
jq '.data.alerts[] |
  select(
    .labels.alertname ==
    "ForecastingPredictionServerErrorRateHigh"
  )'
```

Record the active release lineage:

```bash
curl -s http://localhost:8000/readyz |
jq '{
  release_id,
  model_version,
  model_run_id
}'
```

Search the API logs for prediction failures:

```bash
docker compose logs \
  --since=30m \
  api |
grep -E \
  "Prediction failed|ERROR|Traceback"
```

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
  http://localhost:9090/api/v1/query |
jq .
```

Run a semantic prediction test:

```bash
make predict-test
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
make predict-test
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
  http://localhost:9090/api/v1/query |
jq .
```

### Escalate when

- The 5xx rate remains elevated after a rollback.
- Failures cannot be isolated to specific inputs.
- Training and serving features may be calculated differently.
- Multiple releases produce the same failures.

---

## 7. ForecastingPredictionLatencyHigh

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
  http://localhost:9090/api/v1/query |
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
  http://localhost:9090/api/v1/query |
jq .
```

Inspect container resource usage:

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

### Immediate actions

- Identify large batches or sudden traffic spikes.
- Check CPU and memory utilization.
- Ensure no MLflow or filesystem access occurs in the prediction request path.
- Determine which internal processing stage causes the delay.
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

### Recovery verification

```bash
make predict-test
docker stats --no-stream
```

Observe the p95 latency for at least one complete alert evaluation window.

### Escalate when

- p95 latency remains elevated under normal traffic.
- The API repeatedly reaches memory or CPU limits.
- Horizontal scaling or architectural changes are required.

---

## 8. Incident closure

An incident may be closed when:

- `/livez` and `/readyz` return successful responses.
- `make predict-test` succeeds.
- The active release lineage has been recorded.
- The affected alert has returned to `inactive`.
- Metrics remain stable for at least one complete alert window.

Document:

- Incident start and end time
- Triggered alert
- Release ID before and after remediation
- Root cause
- Actions performed
- Whether a rollback was performed
- Follow-up tasks