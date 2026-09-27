# Sales Forecasting MLOps

Production-oriented sales forecasting system

## Project type

This project implements a production-oriented **forecasting**
machine-learning system.

## Included foundation

The generated project provides a reusable MLOps foundation with:

- environment-specific YAML configuration
- local and Google Cloud Storage support
- immutable serving-release manifests
- active-release pointers and rollback support
- MLflow model loading
- thread-safe model management and reloads
- task-specific serving bundles
- FastAPI inference endpoints
- API-key authentication
- request IDs and safe error responses
- Prometheus metrics
- structured model-lifecycle logging and optional webhook notifications
- privacy-safe prediction event logging
- Docker and Docker Compose support
- automated unit and integration tests
- Rossmann-specific ingestion and schema validation
- stateful time-series feature engineering
- chronological forecasting splits
- XGBoost training with transformed targets
- forecasting evaluation and promotion policy
- immutable serving releases with forecasting artifacts
- partitioned inference monitoring records
- delayed-label rolling performance monitoring
- numeric and categorical feature-drift detection
- policy-controlled automated retraining
- scheduled Prefect auto-retraining deployment
- persisted operational monitoring summary

## Technology stack

- Python 3.12.9
- uv for dependency management
- FastAPI and Uvicorn for model serving
- MLflow for model loading and registry integration
- pandas and PyArrow for tabular data
- fsspec and gcsfs for storage abstraction
- Prometheus for service metrics
- Docker and Docker Compose
- Ruff and pytest for code quality

## Requirements

For local Python development:

- Python 3.12.9
- uv
- GNU Make

For containerized execution:

- Docker
- Docker Compose

## Local setup

Synchronize the project environment:

```bash
make sync
```

`make sync` runs `uv sync`. During the initial setup, it creates the
reproducible uv.lock dependency lock file. Commit this file to version
control before running CI or building the container image.


Run all required quality checks:

```bash
make check
```

The command executes:

```bash
uv run ruff check .
uv run pytest
```

The primary Python package is:

```text
mlops_sales_forecasting
```

## Environment configuration

Create a local environment file:

```bash
cp .env.example .env
```

Change the example API key before using the application outside local
development:

```dotenv
APP_ENV=dev
LOG_LEVEL=INFO
API_PORT=8000
API_KEY=replace-with-a-local-development-key
IMAGE_TAG=local
```

The `.env` file is ignored by Git and must not contain committed secrets.

## Run the API locally

Start the API directly in the uv environment:

```bash
uv run uvicorn mlops_sales_forecasting.api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

The interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## Run with Docker Compose

Validate the Compose configuration:

```bash
make api-config
```

Build and start the API:

```bash
make api-up
```

Show the container status:

```bash
make api-ps
```

Follow the API logs:

```bash
make api-logs
```

Stop the local stack:

```bash
make api-down
```

## Health endpoints

The liveness endpoint confirms that the API process is running:

```bash
curl http://localhost:8000/livez
```

A successful response returns HTTP 200.

The readiness endpoint confirms that a valid serving bundle is loaded:

```bash
curl http://localhost:8000/readyz
```

The readiness endpoint returns HTTP 503 when the API is running but no model
bundle is available. This is expected before the first serving release has
been configured.

## Prediction endpoint

Predictions require the configured API key:

```bash
curl \
  --request POST \
  --header "Content-Type: application/json" \
  --header "X-API-Key: replace-with-a-local-development-key" \
  --data '{
    "inputs": [
      {
        "Store": 1,
        "Date": "2026-09-28",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      }
    ]
  }' \
  http://localhost:8000/predict
```

Each input row represents one store and forecast date. The serving layer
validates the request, joins the active store metadata and known calendar,
injects the latest forecasting state, aligns the generated features with the
MLflow model signature and applies the configured inverse target
transformation.

Unknown stores, invalid dates, missing required fields and incomplete
calendar coverage are rejected with HTTP `422`.

Structured application logs contain privacy-safe operational metadata such
as request ID, release ID, model version, batch size and execution time. They
do not contain raw request records or prediction values.

A separate monitoring store persists only explicitly allowlisted forecasting
features, the store/date matching keys, prediction value, release ID and
request ID. Each request is written as an immutable daily partition under
`data/predictions/history/`. This data supports delayed-label performance and
feature-drift monitoring.

## Metrics

Prometheus-compatible metrics are exposed at:

```text
http://localhost:8000/metrics
```

The metrics include request counts, response status codes and request
latencies.

## Operational monitoring

The API exposes an aggregate monitoring view at:

```text
http://localhost:8000/monitoring/summary
```

The response includes:

- serving readiness and the active release ID;
- the latest rolling RMSE, MAE and forecast bias;
- the most recent feature-drift evaluation;
- the latest persisted automated-retraining state.

Prometheus metrics remain available at `/metrics`. Grafana and Alertmanager
provide service-level visualization and alerting, while the summary endpoint
presents persisted model-operational state.

### Operations dashboard

A Streamlit operations dashboard combines the current serving state with
persisted forecast performance, feature drift, automated-retraining state
and estimated training costs.

Start the complete local monitoring stack:

```bash
make monitoring-up
```

Or start only the API and dashboard:

```bash
make dashboard-up
```

Open the dashboard at:

```text
http://localhost:8501
```

The dashboard and Grafana serve different purposes:

- Streamlit provides a compact model-operations and business-facing overview;
- Grafana visualizes Prometheus time series, service-level objectives and
  alerts;
- MLflow remains the source of experiment and model-run metadata;
- Prefect provides flow-run and deployment visibility.

Training costs are explicitly estimates. They are calculated from completed
MLflow run durations and the configured hourly rate:

```yaml
costs:
  training:
    enabled: true
    currency: EUR
    estimated_hourly_rate: 0.40
    window_days: 30
  scenarios:
    drift_triggered_runs_per_month: 8
```

The resulting report shows observed-window cost estimates and projected
monthly costs for daily, weekly and drift-triggered retraining. It does not
replace provider billing data.

Successful predictions create immutable Parquet files under:

```text
data/predictions/history/date=YYYY-MM-DD/
```

Delayed Ground Truth is supplied through CSV files matching:

```text
data/raw/new_batches/ground_truth_*.csv
```

Each Ground-Truth row must contain at least `Store`, `Date` and `Sales`.
Repeated monitoring refreshes rebuild cumulative Ground Truth from all
available batches and retain the latest value for duplicate `Store` and
`Date` keys.

The refresh produces:

```text
data/monitoring/cumulative_ground_truth.csv
data/monitoring/performance_rolling.parquet
data/monitoring/feature_drift_history.parquet
```

Missing labels or insufficient sample counts are normal bootstrap states and
do not fail the API.

## Automated retraining

The scheduled Prefect deployment evaluates monitoring evidence every day at
03:00 in the `Europe/Berlin` timezone.

Start the local orchestration components:

```bash
make prefect-up

export PREFECT_API_URL=http://127.0.0.1:4200/api

make prefect-pool
make prefect-deploy
```

Start the worker in a separate terminal:

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api

make prefect-worker
```

The automated cycle performs the following operations:

1. rebuild cumulative Ground Truth;
2. refresh rolling forecast performance;
3. evaluate feature drift;
4. validate new Ground-Truth batches;
5. evaluate minimum rows, cooldown and budget limits;
6. evaluate scheduled, performance and drift triggers;
7. train at most one Candidate for a unique decision;
8. run the normal MLflow registration, promotion and serving-release
   lifecycle;
9. persist the completed decision to prevent duplicate retraining.

New data alone does not automatically replace the Champion. Training requires
enough new validated rows and at least one configured trigger. Candidate
promotion remains subject to the normal evaluation and promotion policy.

Do not manually run the `auto-retraining` deployment against production-like
data merely as a connectivity test because it may start a real training
lifecycle.

## Serving releases

A serving release groups the model and all required inference artifacts into
one immutable, validated unit.

The active-release pointer determines which release is loaded by the API.
Replacing the pointer enables controlled promotion and rollback without
mixing artifacts from different model versions.

## Model lifecycle notifications

Training lifecycle events are written to the application logs and can
optionally be delivered to an HTTP webhook.

The following events are available:

- pipeline failure
- candidate rejection by the quality gate
- Challenger registration without promotion
- successful Champion promotion
- successful serving-release publication

Webhook delivery is disabled by default. To enable it, update the appropriate
environment configuration:

```yaml
notifications:
  enabled: true
  log_events: true
  fail_on_error: false
  webhook:
    enabled: true
    url: "${LIFECYCLE_WEBHOOK_URL:-}"
    timeout_seconds: 5.0
```

Provide the URL only through the runtime environment:

```dotenv
LIFECYCLE_WEBHOOK_URL=https://example.com/your-secret-webhook
```

Do not commit webhook URLs because they commonly contain credentials or secret
tokens.

With `fail_on_error: false`, a temporary notification outage is logged but
does not invalidate an otherwise successful training or promotion lifecycle.
Set it to `true` only when notification delivery is a mandatory operational
requirement.

## Project structure

```text
.
├── configs
├── data
│   ├── predictions
│   └── monitoring
├── docs
├── infrastructure
├── monitoring
├── src
│   └── mlops_sales_forecasting
│       ├── api
│       ├── configs
│       ├── data
│       ├── inference
│       ├── monitoring
│       ├── notifications
│       ├── orchestration
│       ├── pipeline
│       ├── storage
│       ├── tracking
│       └── training
├── tests
│   ├── integration
│   └── unit
├── compose.yaml
├── Dockerfile
├── Makefile
├── prefect.yaml
└── pyproject.toml
```

## Project status

This project was generated from the reusable MLOps project template.

Project-specific data ingestion, feature engineering, training, evaluation
and business monitoring must be implemented for the selected use case.


## Cloud deployment

Google Cloud infrastructure and keyless GitHub Actions deployment are
documented in
[docs/cloud-deployment.md](docs/cloud-deployment.md).


## Operations

Operational triage, incident recovery and rollback procedures are
documented in
[docs/operations-runbook.md](docs/operations-runbook.md).


## Template updates

Instructions for applying newer template releases to an existing project are
documented in
[docs/template-updates.md](docs/template-updates.md).