# Local Development

## Purpose

This guide describes how to start, bootstrap, test and reset the complete local
MLOps stack.

## Prerequisites

- Python 3.12;
- `uv`;
- Docker with Docker Compose;
- GNU Make;
- raw Rossmann files under `data/raw/`.

The raw data is intentionally excluded from version control.

## Configure the Project

```bash
git clone https://github.com/SL14-SL/mlops-sales-forecasting.git
cd mlops-sales-forecasting
cp .env.example .env
```

Set a development API key and retain `APP_ENV=dev` for local commands. Local
Make targets explicitly use container-network URLs where required, for example
`http://mlflow:5000` from inside the API container.

Initialize the local Python environment when needed:

```bash
make setup
```

## Start the Stack

```bash
make dev-up
```

Check container state:

```bash
docker compose ps
```

| Service | URL |
|---|---|
| Forecasting API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Prefect | http://localhost:4221 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |


<p align="center">
  <img src="images/swagger_ui.png" width="85%">
</p>

<p align="center">
  <em>FastAPI interface exposing health checks, serving-release administration and authenticated prediction endpoints.</em>
</p>

## Initial Bootstrap

A fresh MLflow registry has no champion. Create the first champion and serving
release with:

```bash
make train-bootstrap
```

Bootstrap is deliberately rejected after a champion exists. Use a regular
training target for all subsequent model updates.

Verify the API:

```bash
curl -fsS http://localhost:8000/livez | jq .
curl -fsS http://localhost:8000/readyz | jq .
make predict-test
```

## Regular Training

Force a candidate run regardless of the pre-training drift check:

```bash
make train-force
```

The candidate must still outperform the champion. `force` bypasses the decision
to skip training; it does not bypass the promotion gate.

## Prefect Deployment and Worker

Register or update the scheduled auto-retraining deployment:

```bash
make prefect-setup
```

Start a local process worker in a separate terminal:

```bash
make prefect-worker
```

Run the decision flow once without waiting for its schedule:

```bash
make auto-retrain
```

The worker process must remain active for scheduled deployment runs.

## Serving Release Operations

List available releases:

```bash
make list-serving-releases
```

Use the project rollback target to activate a previous release. Check the
Makefile help for its required release argument:

```bash
make help
```

After activation, verify that `/readyz` reports the expected release ID and run
`make predict-test`.

## Monitoring Checks

Confirm that the API exposes a ready serving bundle:

```bash
curl -fsS http://localhost:8000/metrics \
  | grep -A2 api_serving_ready
```

Query Prometheus:

```bash
curl -sG \
  --data-urlencode 'query=api_serving_ready' \
  http://localhost:9090/api/v1/query \
  | jq .
```

Validate alerting configuration:

```bash
docker compose exec -T prometheus \
  promtool check rules /etc/prometheus/alerts.yml

docker compose run --rm \
  --no-deps \
  --entrypoint /bin/amtool \
  alertmanager \
  check-config /etc/alertmanager/alertmanager.yml
```

## Quality Checks

```bash
pytest
ruff check .
docker compose config --quiet
```

Project Make targets may wrap the first two commands:

```bash
make test
make lint
```

## Troubleshooting

### API is alive but not ready

Inspect the readiness response and API logs:

```bash
curl -i http://localhost:8000/readyz
docker compose logs --tail=200 api
```

Typical causes are a missing release, inaccessible MLflow artifact or failed
serving-bundle validation.

### Prefect client/server version warning

Keep the Prefect image version aligned with the package version in the project
lock file. Recreate the server and worker after changing the version.

### Host and container URLs

Use `localhost` from the host and Docker service names from containers:

| Caller | MLflow URL | Prefect URL | API URL |
|---|---|---|---|
| Host | `http://localhost:5000` | `http://localhost:4221/api` | `http://localhost:8000` |
| Container | `http://mlflow:5000` | `http://prefect:4200/api` | `http://api:8080` |

### Clean reset

A complete demo reset deletes local model, registry and lifecycle state. Use
only the dedicated Make target documented by `make help`, and verify its scope
before execution. Do not use broad recursive deletion commands manually.

## Related Documentation

- [Architecture](architecture.md)
- [Serving releases](serving-releases.md)
- [Retraining policy](retraining-policy.md)
- [Monitoring and SLOs](monitoring-and-slos.md)

