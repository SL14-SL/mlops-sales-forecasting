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
  --data '{"inputs": []}' \
  http://localhost:8000/predict
```

The exact request fields depend on the selected project type. Empty inputs
are rejected with a validation error.

Prediction event logs contain operational metadata such as request ID,
release ID, model version, batch size and execution time. Raw input features
and prediction values are not written to these technical logs.

## Metrics

Prometheus-compatible metrics are exposed at:

```text
http://localhost:8000/metrics
```

The metrics include request counts, response status codes and request
latencies.

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
├── src
│   └── mlops_sales_forecasting
│       ├── api
│       ├── configs
│       ├── inference
│       │   └── releases
│       ├── monitoring
│       ├── storage
│       └── utils
├── tests
│   ├── integration
│   └── unit
├── Dockerfile
├── Makefile
├── compose.yaml
├── pyproject.toml
└── uv.lock
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