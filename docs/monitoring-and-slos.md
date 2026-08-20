# Monitoring, SLOs and Alerting

## Purpose

The platform monitors two different kinds of production risk:

1. operational reliability of the prediction service;
2. statistical quality of the forecasts after ground truth becomes available.

Both are required. A fast API can serve a poor model, while an accurate model
is not useful if the API is unavailable.

## Terminology

- **SLI — Service Level Indicator:** the measured signal, such as success rate
  or p95 latency.
- **SLO — Service Level Objective:** the target for an SLI, such as at least
  99% successful prediction requests.
- **Alert:** a notification condition indicating that an objective is at risk
  or already violated.

SLOs are internal engineering objectives, not contractual customer guarantees.

## Operational Metrics

The FastAPI service exposes Prometheus metrics at `/metrics`.

Important signals include:

| Signal | Purpose |
|---|---|
| `api_serving_ready` | Whether a complete serving bundle is active |
| Request count | Traffic volume by path and status |
| Response-status count | 2xx, 4xx and 5xx behavior |
| Request-duration histogram | p50, p95 and p99 latency |
| Prediction throughput | Requests per minute |

The deployment prediction probe is real inference traffic but is tagged so it
does not contaminate normal prediction-history based ML metrics.

## Example SLOs

Concrete thresholds are configured in the monitoring files. The following
table describes their intent:

| Objective | Indicator | Expected behavior |
|---|---|---|
| API availability | Prometheus scrape/up state | API remains reachable |
| Serving readiness | `api_serving_ready` | A complete bundle stays active |
| Prediction reliability | 5xx ratio | Server-error rate remains low |
| Prediction latency | p95 duration | Prediction latency remains below its limit |

Client-side 4xx responses are not server failures and should not be counted as
5xx error-rate breaches.

## Alert Rules

The core rules cover:

- `ForecastingAPIUnavailable`;
- `ForecastingServingBundleNotReady`;
- `ForecastingPredictionServerErrorRateHigh`;
- `ForecastingPredictionLatencyHigh`.

Rules should include a `for` duration where appropriate so a single scrape or
short startup transition does not page an operator.

Validate the rules:

```bash
docker compose exec -T prometheus \
  promtool check rules /etc/prometheus/alerts.yml
```

Inspect loaded rules:

```bash
curl -s http://localhost:9090/api/v1/rules \
  | jq '.data.groups[].rules[] | {name, state, health}'
```

## Alertmanager

Prometheus forwards firing alerts to Alertmanager. Alertmanager handles:

- grouping;
- repeat intervals;
- receiver routing;
- optional external delivery such as Slack.

Validate configuration:

```bash
docker compose run --rm \
  --no-deps \
  --entrypoint /bin/amtool \
  alertmanager \
  check-config /etc/alertmanager/alertmanager.yml
```

Verify Prometheus connectivity:

```bash
curl -s http://localhost:9090/api/v1/alertmanagers | jq .
```

The local alert receiver records delivery attempts when no external Slack
configuration is present. A skipped external delivery is expected in that
configuration and does not mean that Prometheus-to-Alertmanager routing failed.

## Grafana

The operational dashboard presents:

- serving readiness;
- API success rate;
- prediction server-error rate;
- latency percentiles;
- HTTP status distribution;
- request throughput.

`No data` does not always indicate failure. Rate expressions require matching
counter series and enough samples within their lookback window. Queries should
use the metric labels actually exported by the API.

<p align="center">
  <img src="images/grafana_dashboard_slo.png" width="100%">
</p>

<p align="center">
  <em>Production SLO dashboard showing a ready serving bundle, 100% availability, 115 ms p95 latency and no observed server errors during the measurement window.</em>
</p>

## ML Quality Monitoring

### Data quality

Runtime checks record:

- row count;
- missing rates for required fields;
- unseen categories;
- validation status and reason.

### Feature drift

Selected input features are compared with reference distributions. The
retraining policy requires persistence across consecutive windows rather than
reacting to a single drift result.

### Forecast performance

Predictions are produced before actual sales become available. Later,
ground-truth batches are joined to prediction history and rolling metrics are
calculated:

- RMSE;
- MAE;
- bias.

Missing join matches are reported explicitly. They do not count as performance
degradation.

## Dashboard Roles

Grafana and Streamlit serve different audiences:

| Dashboard | Primary purpose |
|---|---|
| Grafana | Live API reliability and SLO status |
| Streamlit | Forecast quality, lifecycle evidence and controlled experiment comparison |

## Verification Commands

Check current API readiness:

```bash
curl -fsS http://localhost:8000/readyz | jq .
```

Inspect the readiness metric directly:

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

## Initial Incident Response

| Alert | First checks |
|---|---|
| API unavailable | Container/Cloud Run status, `/livez`, application logs |
| Bundle not ready | `/readyz`, active release pointer, artifact access and API logs |
| High 5xx rate | Recent releases, prediction exceptions and request samples |
| High latency | Request volume, p95/p99 trend, CPU/memory and feature processing |

If an incident begins immediately after release activation, compare the current
release ID with the previous release and use the tested rollback procedure when
verification did not already restore it.

## Related Documentation

- [Architecture](architecture.md)
- [Serving releases](serving-releases.md)
- [Retraining policy](retraining-policy.md)
- [Production demo](production-demo.md)
