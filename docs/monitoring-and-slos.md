# Monitoring, SLOs and Alerting

## Purpose

This document describes the observability model for the Sales Forecasting
MLOps system.

The monitoring design separates:

- service health and request behavior;
- serving-bundle readiness;
- prediction execution;
- forecast quality after delayed labels arrive;
- feature drift;
- automated-retraining decisions;
- estimated training costs.

No single metric is treated as sufficient evidence for retraining or model
promotion.

## Monitoring Layers

| Layer | Source | Primary consumer |
|---|---|---|
| Service metrics | FastAPI middleware and Prometheus client | Prometheus and Grafana |
| Serving readiness | `ModelManager` and `/readyz` | Prometheus alerts and operators |
| Prediction metrics | Prediction router and model observability | Grafana dashboards |
| Inference history | Allowlisted immutable Parquet partitions | Monitoring refresh |
| Forecast performance | Predictions joined with delayed ground truth | Retraining policy and dashboard |
| Feature drift | Reference and recent inference features | Retraining policy and dashboard |
| Retraining state | Persisted policy decisions | API summary and Streamlit |
| Training costs | Completed MLflow runs and configured hourly rate | Streamlit dashboard |

## Operational Metrics

The API exposes Prometheus-compatible metrics at:

```text
http://localhost:8000/metrics
```

Prometheus scrapes the API every 15 seconds.

Important service metrics include:

- total HTTP requests by status code;
- request latency histogram;
- request rate;
- server-error ratio;
- availability ratio;
- p95 latency;
- latency-SLI ratio;
- remaining error-budget ratio;
- serving readiness.

Prediction-specific metrics include:

- prediction request rate;
- prediction observation rate;
- prediction error ratio;
- p95 prediction latency;
- forecast value distribution;
- requested forecast-horizon distribution;
- forecast MAE, RMSE and bias after feedback;
- number of features currently marked as drifted.

Recording rules are defined in:

```text
monitoring/prometheus/recording_rules.yml
```

## Service-Level Objectives

The provided recording rules model these example objectives:

| Objective | Target |
|---|---:|
| Successful API responses | At least `99.5%` |
| Requests completed within one second | Measured as a rolling five-minute ratio |
| API p95 request latency | At most `1 second` |
| Prediction error ratio | At most `5%` |
| Prediction p95 latency | At most `1 second` |
| Serving readiness | One complete serving bundle must be active |

The error-budget recording rule uses an allowed server-error ratio of `0.5%`.
These are demonstration SLOs and must be calibrated against real traffic,
business impact and infrastructure capacity before production use.

## Alert Rules

Prometheus evaluates the following alerts:

| Alert | Condition | Duration | Severity |
|---|---|---:|---|
| `MlopsApiDown` | API scrape target is unavailable | 2 minutes | critical |
| `MlopsApiHighServerErrorRate` | More than 5% server errors with meaningful traffic | 5 minutes | warning |
| `MlopsApiHighP95Latency` | API p95 latency exceeds 1 second with meaningful traffic | 5 minutes | warning |
| `PredictionErrorRateHigh` | Prediction error ratio exceeds 5% | 10 minutes | warning |
| `PredictionLatencyHigh` | Prediction p95 latency exceeds 1 second | 10 minutes | warning |
| `MlopsServingBundleNotReady` | API is reachable but no complete bundle is ready | 2 minutes | critical |

Alert definitions are stored in:

```text
monitoring/prometheus/alerts.yml
monitoring/prometheus/serving_readiness.yml
```

## Alertmanager

Alertmanager is available locally at:

```text
http://localhost:9093
```

The local configuration:

- groups alerts by alert name, service and severity;
- waits 15 seconds before sending the first grouped notification;
- groups subsequent updates for five minutes;
- repeats unresolved alerts every four hours;
- suppresses matching warning alerts while a critical alert is active.

The default `local-alerts` receiver does not send notifications externally.
Production environments should configure an approved webhook, email or
incident-management receiver.

The configuration is stored in:

```text
monitoring/alertmanager/alertmanager.yml
```

## Grafana

Grafana is available locally at:

```text
http://localhost:3000
```

The default credentials come from these environment variables:

```text
GRAFANA_ADMIN_USER
GRAFANA_ADMIN_PASSWORD
```

Prometheus is provisioned automatically as the default data source.

The repository contains four dashboards:

| Dashboard | Purpose |
|---|---|
| `api-overview.json` | API request volume, errors and latency |
| `slo-overview.json` | Availability, latency SLI and error budget |
| `prediction-overview.json` | Prediction traffic, failures and execution latency |
| `model-observability.json` | Forecast quality, output distributions and feature drift |

Dashboard definitions are stored under:

```text
monitoring/grafana/dashboards/
```

## Streamlit Operations Dashboard

The operations dashboard is available at:

```text
http://localhost:8501
```

It combines persisted model-operational state that is not naturally represented
by short-lived Prometheus series:

- serving readiness and active release;
- latest rolling forecast performance;
- feature-drift results;
- latest automated-retraining decision;
- MLflow-based training-cost estimates;
- monthly training-cost scenarios.

The second Streamlit page visualizes the lifecycle simulation and compares the
static model with the policy-controlled retraining scenario.

## Inference History

Successful predictions write an explicitly allowlisted monitoring record. The
technical application logs do not contain raw request records or prediction
values.

Inference history is partitioned by date under:

```text
data/predictions/history/date=YYYY-MM-DD/
```

Records include only the fields required for delayed-label evaluation and
feature-drift analysis, such as:

- request and release identifiers;
- store and forecast date;
- prediction value;
- selected monitored features;
- event timestamp.

The write path is intentionally fail-safe by default. A monitoring-storage
failure is logged but does not invalidate an otherwise successful prediction
unless `monitoring.inference_logging.fail_on_error` is enabled.

## Delayed Ground Truth

Ground-truth batches are supplied as CSV files matching:

```text
data/raw/new_batches/ground_truth_*.csv
```

Each row must contain at least:

- `Store`;
- `Date`;
- `Sales`.

Repeated refreshes rebuild cumulative ground truth from all available batches.
If duplicate `Store` and `Date` keys exist, the latest available value is
retained.

The refresh produces:

```text
data/monitoring/cumulative_ground_truth.csv
data/monitoring/performance_rolling.parquet
data/monitoring/feature_drift_history.parquet
```

## Forecast Performance

Predictions are joined with delayed ground truth by the configured forecasting
keys. The monitoring layer calculates rolling:

- root mean squared error;
- mean absolute error;
- forecast bias;
- evaluated sample count.

The development retraining policy uses a seven-day rolling window with at
least 500 matched samples.

Configured degradation limits are:

| Metric | Limit |
|---|---:|
| RMSE | `1375.0` |
| MAE | `990.0` |
| Absolute bias | `900.0` |

Performance degradation must persist for two consecutive windows before it is
treated as retraining evidence.

## Feature Drift

Feature drift compares recent inference features with the validated training
reference.

The development configuration monitors:

### Numeric features

- `CompetitionDistance`.

Numeric drift uses the two-sample Kolmogorov-Smirnov test. Drift is detected
only when both conditions hold:

- p-value is below `0.01`;
- KS statistic is above `0.10`.

### Categorical features

- `Promo`;
- `Store`;
- `StoreType`;
- `Assortment`;
- `StateHoliday`.

Categorical drift uses a chi-square goodness-of-fit comparison with a p-value
threshold of `0.01`.

Both reference and current samples require at least 50 valid observations.
The current inference lookback is 14 days.

A single drift result is not sufficient for automatic retraining. Drift for
the same feature must persist for two consecutive evaluation windows.

## Retraining Evidence

The signal collector combines monitoring evidence with lifecycle state.

The development policy uses:

| Setting | Value |
|---|---:|
| Minimum new training rows | `500` |
| Maximum new training rows | `1,000,000` |
| Cooldown | `168 hours` |
| Scheduled refresh interval | `168 hours` |
| Drift lookback | `14 days` |
| Consecutive drift windows | `2` |
| Consecutive degraded performance windows | `2` |

The policy can return `skip`, `train_candidate` or `block`. Training a
candidate does not imply promotion. The independent promotion policy still
decides whether the challenger becomes the active champion.

See [Retraining policy](retraining-policy.md) for the complete decision order.

## Monitoring Summary API

The aggregate operational view is exposed at:

```text
GET /monitoring/summary
```

Example command:

```bash
curl \
  --fail \
  --silent \
  --show-error \
  http://localhost:8000/monitoring/summary
```

The response contains:

- generation timestamp;
- serving readiness;
- active release ID;
- latest reload error;
- latest rolling performance;
- latest drift evaluation;
- latest persisted retraining state.

Unavailable histories are reported explicitly rather than represented as
fabricated zero values.

## Related Documentation

- [System architecture](architecture.md)
- [Local development](local-development.md)
- [Retraining policy](retraining-policy.md)
- [Serving releases](serving-releases.md)
- [Operations runbook](operations-runbook.md)
