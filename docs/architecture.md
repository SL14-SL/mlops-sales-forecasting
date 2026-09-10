# System Architecture

## Purpose

This document describes the runtime components, ownership boundaries and data
flows of the sales-forecasting platform. The architecture is designed to keep
training, model selection, release publication and online serving independently
testable.

## High-Level Architecture

```mermaid
flowchart TD
    A["Raw sales and store data"] --> B["Validation and features"]
    B --> C["Chronological dataset snapshot"]
    C --> D["Prefect training flow"]
    D --> E["Candidate evaluation"]
    E --> F["MLflow registry"]
    F --> G["Immutable GCS release"]
    G --> H["Active release pointer"]
    H --> I["FastAPI serving"]
    I --> J["Verification and monitoring"]
    J -->|Signals| D
    J -->|Verification failure| K["Rollback"]
```

<p align="center">
  <img src="images/prefect_flow_overview.png" width="100%">
</p>

<p align="center">
  <em>Completed end-to-end forecasting pipeline covering data processing, model training, champion registration, serving-release publication and semantic verification.</em>
</p>


## Component Responsibilities

| Component | Responsibility | Persistent information |
|---|---|---|
| Data pipeline | Validation, temporal features and state updates | Processed data and feature state |
| Prefect | Training and retraining orchestration | Flow and task run metadata |
| MLflow | Experiments, runs, metrics and model versions | Cloud SQL metadata and GCS artifacts |
| Cloud SQL | Persistent PostgreSQL backend for MLflow | Experiments, runs, registry metadata and aliases |
| GCS | MLflow artifacts, dataset snapshots and immutable serving releases | Versioned objects |
| Secret Manager | Supplies the MLflow database password | Versioned secret |
| FastAPI | Request validation and prediction serving | Process-local active bundle; persistent authority remains the release pointer |
| Prometheus | Metric collection and alert-rule evaluation | Time-series metrics |
| Grafana | Operational visualization | Dashboard definitions |
| Alertmanager | Alert grouping and routing | Alert state |
| Terraform | Cloud resource definition | Terraform state |
| GitHub Actions | Validation, image publication and deployment | Workflow history |

## Internal Code Boundaries

The application code is separated by responsibility so that orchestration,
transport concerns and domain logic remain independently testable.

| Layer | Main modules | Responsibility |
|---|---|---|
| Flow orchestration | `flows/training_flow.py`, `flows/auto_retrain_flow.py` | Coordinate lifecycle steps without implementing task internals |
| Prefect tasks | `flows/tasks/` | Data preparation, training, registry and serving tasks |
| Deployment orchestration | `flows/deployment_flow.py` | API reload, semantic verification and automatic rollback |
| HTTP transport | `src/api/app.py`, `src/api/routers/` | FastAPI assembly, routing, authentication and status codes |
| API request handling | `src/api/prediction_handler.py` | Data-quality logging, prediction logging and response construction |
| Serving state | `src/api/serving_state.py` | Atomically activate and expose one process-local serving bundle |
| Inference execution | `src/inference/prediction_service.py` | Feature preparation, model execution and prediction postprocessing |
| Release lifecycle | `src/inference/releases/` | Manifest handling, storage, publication and active-pointer operations |
| Training lifecycle | `src/training/` | Dataset preparation, weighting, training, comparison, final refit and metadata |
| Storage abstraction | `src/storage/` | Local and object-storage filesystem operations |

## Training and Promotion Flow

```mermaid
flowchart TD
    A["Prepare chronological data"] --> B["Train candidate"]
    B --> C["Evaluate candidate and champion"]
    C --> D{"Candidate better?"}
    D -->|No| E["Keep champion"]
    D -->|Yes| F["Final refit"]
    F --> G["Register model version"]
    G --> H["Assign champion alias"]
    H --> I["Publish serving release"]
    I --> J["Reload and verify API"]
```

The candidate and champion are evaluated on the same chronological validation
data and on the original target scale. An accepted candidate is not served
directly: a separate final model is refitted on the combined training and
validation data.

## Serving Architecture

MLflow and GCS have different responsibilities:

- MLflow is the source of truth for training runs and registered model versions.
- A serving release defines the exact combination of model and inference assets.
- The active pointer selects one complete release.
- The API changes its in-memory state only after the candidate bundle has loaded
  and validated successfully.

Within the API process, `src.api.serving_state` is the only owner of the active
bundle reference. Health, readiness, metrics and prediction endpoints all read
that same reference. This prevents stale copies of the serving state across
different routers.

This prevents combinations such as new model weights with stale forecasting
state or calendar data.

## Environment Topology

### Local development

Docker Compose starts:

- FastAPI;
- PostgreSQL;
- MLflow;
- Prefect server and optional worker;
- Streamlit;
- Prometheus;
- Grafana;
- Alertmanager and the local alert receiver.

PostgreSQL provides a persistent local MLflow backend. Repository directories
are mounted for data, models, monitoring output and serving releases.

### Google Cloud demonstration

The cloud demonstration uses:

- Cloud Run for MLflow and the forecasting API;
- Cloud SQL for PostgreSQL as the durable MLflow tracking backend;
- Secret Manager for the database password;
- Artifact Registry for container images;
- GCS for raw data, MLflow artifacts, dataset snapshots and serving releases;
- Terraform for resource provisioning;
- GitHub Actions with Workload Identity Federation for deployment.

MLflow stores experiments, runs, registered-model metadata and aliases in Cloud
SQL. Large model artifacts remain in GCS. This separation keeps MLflow metadata
persistent across Cloud Run instance termination, scale-to-zero and revision
replacement.

The Terraform configuration limits the MLflow Cloud Run service to one instance
and permits scale-to-zero. Cloud SQL remains the main continuously billable
resource and is only provisioned for the duration of the demonstration.

<p align="center">
  <img src="images/gcp_cloud_run_overview.png" width="70%">
</p>

<p align="center">
  <em>Google Cloud Run services hosting MLflow and the production forecasting API.</em>
</p>

<p align="center">
  <img src="images/cloud_run_mlflow_cloud_sql.png" width="100%">
</p>

<p align="center">
  <em>Persistent MLflow architecture using Cloud Run, Cloud SQL for PostgreSQL, Secret Manager and GCS artifact storage.</em>
</p>

## Trust Boundaries

| Boundary | Control |
|---|---|
| Client to API | API key and schema validation |
| GitHub to GCP | Workload Identity Federation |
| API to release storage | Dedicated service account and bucket IAM |
| Release activation | Manifest validation and artifact checksums |
| Deployment completion | Readiness and semantic prediction verification |
| Failed deployment | Pointer restoration and API reload |

## Reusable and Domain-Specific Layers

Reusable infrastructure includes orchestration, registry integration, serving
releases, health checks, monitoring, rollback, CI/CD and Terraform.

Domain-specific code includes the request schema, feature transformations,
store-level state, target transformation, evaluation metrics and retraining
thresholds. These parts must be adapted when the blueprint is transferred to a
different forecasting problem.

## Related Documentation

- [Local development](local-development.md)
- [Production demo](production-demo.md)
- [Serving releases](serving-releases.md)
- [Retraining policy](retraining-policy.md)
- [Monitoring and SLOs](monitoring-and-slos.md)

