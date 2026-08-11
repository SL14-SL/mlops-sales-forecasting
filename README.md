# 🚀 Production-Oriented MLOps Blueprint for Sales Forecasting

End-to-end MLOps showcase for deploying, monitoring and continuously improving forecasting models in a cloud-native production-style environment.

Sales forecasting is used as the example use case, but the architecture is designed around reusable MLOps patterns: forecast serving, experiment tracking, model registry workflows, monitoring, automated retraining, CI/CD and infrastructure-as-code.

The focus is not only model training, but the engineering layer required to operate forecasting systems reliably after the model has been trained.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Forecasting_API-green)
![MLflow](https://img.shields.io/badge/MLflow-Model_Registry-blue)
![Prefect](https://img.shields.io/badge/Prefect-Orchestration-purple)
![Terraform](https://img.shields.io/badge/Terraform-IaC-623CE4)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-black)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 🧩 Blueprint Positioning

This repository is the forecasting variant of a reusable MLOps blueprint. A
corresponding classification implementation is available in the
[Customer Churn MLOps project](https://github.com/SL14-SL/mlops-churn-prediction).

The goal is not to optimize one specific dataset, but to show how the same production-oriented ML architecture can be adapted to different ML problem types.

| Project | Problem Type | Use Case | Main Adapted Components |
|---|---|---|---|
| Customer Churn MLOps | Binary Classification | Retention risk prediction | Classification metrics, churn decision logic, delayed labels |
| Sales Forecasting MLOps | Time Series / Regression | Demand prediction | Temporal features, forecasting state, regression monitoring |

The shared lifecycle is: data validation, feature engineering, training, MLflow tracking and registry, API serving, prediction logging, monitoring, retraining and CI/CD deployment.

---

# 🏗️ Architecture Overview

The platform implements a complete operational ML lifecycle for sales forecasting.

```mermaid
flowchart TB
A[Raw Sales Data] --> B[Validation]
B --> C[Feature Engineering]
C --> D[Temporal Splitting]
D --> E[Training Pipeline - Prefect]
E --> F[MLflow Tracking]
F --> G[MLflow Model Registry]
G --> H[FastAPI Forecasting API]

H --> I[Forecast Logs]
I --> J[Monitoring Layer]

J --> K[Data Quality Checks]
J --> L[Feature Drift Detection]
J --> M[Forecast Performance Monitoring]
J --> N[Retraining Trigger]

N --> O[Retraining Pipeline]
O --> E
```

The architecture separates reusable MLOps infrastructure from
forecasting-specific logic. 
Validation, orchestration, experiment tracking,
model registration, serving, monitoring and deployment form the reusable
platform layer.

Temporal features, store-level forecasting state, regression evaluation and
demand-oriented inference represent the use-case-specific forecasting layer.

---

# 🔌 API & Forecast Serving

The project exposes a FastAPI service for online forecasting.

The API supports:

- single forecast requests
- batch forecast requests
- model readiness and liveness checks
- active model metadata
- model reload endpoint
- API authentication
- structured forecast responses
- request metadata and timing information

The serving layer loads the active champion model from MLflow and applies the required forecasting feature logic before generating predictions.

<p align="center">
  <img src="docs/images/swagger_ui.png" width="100%">
</p>

<p align="center">
  <em>FastAPI Swagger UI with health, admin reload and forecast prediction endpoints.</em>
</p>

---

# 📈 Forecasting-Specific Inference Layer

Forecasting problems require more than simply passing a row of features into a model.

This project includes a dedicated inference layer for forecasting-specific concerns:

- forecast input contracts
- feature dispatch logic
- temporal feature generation
- store-level metadata handling
- forecasting state loading
- reusable inference context
- single and batch prediction handling

The goal is to keep forecasting logic explicit and testable instead of hiding it inside the API endpoint.

This makes the project easier to extend to other demand prediction or time-dependent regression use cases.

---

# 🔁 Automated Training & Retraining

Training and retraining workflows are orchestrated with Prefect.

The pipeline automates:

- drift checks before training
- raw data ingestion and validation
- feature processing and feature state updates
- dataset snapshotting
- model training
- model evaluation and registration
- MLflow logging
- API refresh after model promotion
- retraining trigger evaluation

In the controlled lifecycle demo, retraining is triggered by persistent
forecast-performance degradation. The broader monitoring architecture also
provides signals for feature drift and data-quality issues, which can be
connected to additional retraining policies.

```mermaid
flowchart LR
A[Monitoring Signal] --> B[Retraining Decision]
B --> C[Train Candidate]
C --> D[Evaluate on Validation Data]
D --> E{Better than Champion?}
E -->|No| F[Keep Champion]
E -->|Yes| G[Final Refit on Train and Validation]
G --> H[Register New Champion]
H --> I[Reload API]
```
The candidate is first trained on the training split and compared with the
current champion on untouched chronological validation data. If the candidate
wins, a separate final model is refitted on both training and validation data.
Only the accepted final-refit model receives the champion alias and is loaded
by the serving API.

During drift retraining, recent promotional observations receive higher sample
weights. The current policy uses weights of 5, 3 and 1.5 for promotional
observations from the most recent 30, 60 and 120 days. All other observations
retain the default weight of 1.

<p align="center">
  <img src="docs/images/prefect_flow.png" width="100%">
</p>

<p align="center">
  <em>Prefect flow run for the end-to-end demand forecasting pipeline, including drift checks, feature processing, model training, evaluation, registration and API refresh.</em>
</p>  

---

# 🧪 Time-Aware Model Development

Forecasting models must be evaluated without using future observations during
training. The project therefore includes walk-forward backtesting with
chronologically expanding training windows and later validation periods.

The model-development process included:

- forecasting-specific calendar and holiday features
- lag and rolling demand features
- chronological walk-forward validation
- Optuna-based XGBoost hyperparameter tuning
- evaluation on the original sales scale
- RMSE, MAE, WMAPE, RMSPE and bias reporting

Walk-forward mean RMSE improved across the model-development stages:

| Model stage | Mean RMSE |
|---|---:|
| Initial baseline | 792.37 |
| Calendar features | 748.31 |
| Tuned XGBoost model | 690.49 |

Compared with the initial baseline, the tuned model reduces mean walk-forward
RMSE by approximately 12.9%.

Run the backtest with:

```bash
uv run python scripts/run_model_backtest.py \
  --output-directory results/model_backtest
```

Hyperparameter studies can be executed with:

```bash
uv run python scripts/tune_model.py \
  --trials 20 \
  --folds 2 3
```

To keep the optimization runtime manageable, hyperparameter tuning is performed
on walk-forward folds 2 and 3. The selected configuration is subsequently
reported across all four folds.

Because folds 2 and 3 also participate in hyperparameter selection, the
four-fold result should be interpreted as a validation estimate rather than
performance on a completely untouched final test period.

---

# 📊 Experiment Tracking & Model Evaluation

MLflow is used for experiment tracking, artifact logging and model registry workflows.

Tracked information includes:

- model parameters
- training metrics
- validation metrics
- evaluation reports
- feature schema artifacts
- model artifacts
- dataset and run metadata
- estimated training costs

Forecasting evaluation focuses on regression-oriented quality metrics such as forecast error and model stability over time.

The Model Registry enables controlled promotion of new model versions and supports reproducible deployment workflows.

<p align="center">
  <img src="docs/images/mlflow_run_overview.png" width="100%">
</p>

<p align="center">
  <em>MLflow run overview with tracked parameters, training metrics, cost estimates and registered model artifacts.</em>
</p>

---

# 🏆 Model Registry & Promotion Workflow

Models are versioned and promoted through the MLflow Model Registry.

The project supports:

- registered model versions
- champion model selection
- reproducible model artifacts
- deployment metadata
- registry-based model loading
- API reload without rebuilding the image

The serving API loads the currently active model from the registry.

This creates a clean separation between:

- training a model
- registering a model
- promoting a model
- serving a model

That separation is an important production MLOps pattern because it allows controlled model updates without tightly coupling training and serving.

<p align="center">
  <img src="docs/images/mlflow_model_details.png" width="100%">
</p>

<p align="center">
  <em>MLflow Model Registry with versioned forecasting models and a champion alias for controlled production serving.</em>
</p>

---

# 📡 Monitoring & Observability

The project includes monitoring capabilities for both ML quality and operational service health.

Monitoring features include:

- feature drift detection
- data quality checks
- prediction logging
- forecast performance tracking
- retraining trigger evaluation
- API request metrics
- latency monitoring
- service health checks
- Prometheus metrics endpoint
- Grafana dashboard support

The monitoring layer evaluates whether the currently deployed champion model still satisfies expected production quality requirements.

Operational API metrics are collected through Prometheus and visualized in Grafana, including success rate, prediction latency, status codes and request throughput.

<p align="center">
  <img src="docs/images/grafana_dashboard.png" width="100%">
</p>

<p align="center">
  <em>Grafana dashboard for operational API monitoring with success rate, p95 prediction latency, HTTP status codes and prediction request throughput.</em>
</p>

---

# 📉 Forecast Performance Monitoring

Forecasting systems are vulnerable to changing demand patterns, seasonality shifts, promotions, holidays and store-level behavior changes.

The project therefore includes monitoring logic for:

- prediction history
- ground-truth comparison
- forecast error tracking
- performance degradation
- retraining decision support

This mirrors a common real-world forecasting setup: predictions are generated first, while actual sales values become available later and can then be used to evaluate model quality.

## Production Performance Monitoring

The first dashboard view presents the operational monitoring state of one
selected lifecycle run. It shows rolling RMSE, MAE and bias, the configured
drift ramp-up period, the retraining trigger and the final-refit champion
promotion.

Lifecycle result files are discovered automatically under `results/`. The
dashboard therefore supports completed and partially running simulations
without requiring manual file copies or changes to the dashboard code.

<p align="center">
  <img src="docs/images/streamlit_dashboard.png" width="90%">
</p>

<p align="center">
  <em>
    Production monitoring view for the mild recency-weighted lifecycle run.
    The dashboard displays rolling forecast metrics, the controlled drift
    period, the retraining event and final-refit champion promotion.
  </em>
</p>

---

## Interactive Retraining Comparison

The second dashboard view compares matching lifecycle runs with and without
automated retraining. Both variants use the same drift scenario, simulation
parameters, initial champion and ground truth.

The interactive selectors make it possible to compare different preserved
lifecycle runs without regenerating dashboard-specific input files.

<p align="center">
  <img
    src="docs/images/streamlit_retraining_comparison.png"
    width="90%"
  >
</p>

<p align="center">
  <em>
    Interactive rolling-RMSE comparison between the static no-retraining
    baseline and the adaptive pipeline using mild recency weighting and a
    final refit.
  </em>
</p>

---

## Controlled Retraining Experiment

A controlled concept-drift experiment evaluates whether automated retraining
improves forecast quality after the relationship between promotions and sales
changes.

The simulation gradually reduces the sales effect of promotions by 25%:

- drift starts on simulation day 20
- full drift strength is reached after 14 days
- retraining is triggered on day 40
- recent promotional observations receive recency weights of 5, 3 and 1.5
- the accepted candidate is refitted on all available training and validation
  observations before promotion
- both variants use identical ground truth and the same initial champion model

### Reproducing the experiment

After training the initial champion, create a reproducible experiment snapshot:

```bash
make snapshot-demo-baseline
```
Run both lifecycle variants and generate the final comparison:

```bash
make controlled-retraining-experiment
```
The experiment target automatically:

- restores the same training data, feature state, simulation pool and champion
  before each variant
- executes one run without retraining
- executes one run with automated retraining and final refitting
- preserves row-level predictions and ground truth for both variants
- generates the rolling and segment-level comparison

This ensures that differences between both variants can be attributed to the
retraining policy rather than different initial conditions.


The interactive dashboard focuses on lifecycle monitoring and rolling metrics.
For the final offline model-quality evaluation, preserved prediction and
ground-truth records are evaluated at row level and segmented into all open,
promotional and non-promotional stores.


<p align="center">
  <img src="docs/images/promo_mild_weights_comparison.png" width="100%">
</p>

<p align="center">
  <em>
    Final controlled evaluation. The upper panel shows rolling RMSE over time,
    while the lower panel compares post-promotion row-level RMSE across
    business-relevant forecast segments.
  </em>
</p>

The dashboard chart reports a rolling lifecycle metric over the most recent
monitoring window. The table below reports row-level metrics across the complete
post-promotion evaluation period. The values therefore answer different
questions and are not expected to be identical.


### Post-promotion results

| Segment | Without retraining RMSE | With final-refit retraining RMSE | Relative RMSE change |
|---|---:|---:|---:|
| All open stores | 915.73 | 893.48 | -2.43% |
| Promotional stores | 1,032.02 | 969.38 | -6.07% |
| Non-promotional stores | 824.74 | 836.33 | +1.41% |

A negative relative RMSE change indicates lower forecast error and therefore
better performance. A positive value indicates higher forecast error.

Across all open stores, final-refit retraining reduces RMSE by 2.43%.
For promotional observations, which are directly affected by the simulated
concept drift, RMSE improves by 6.07%. The non-promotional segment becomes
1.41% worse, demonstrating a measurable trade-off between adaptation to the
changed promotion effect and stability in unaffected observations.

Additional post-promotion metrics also improve overall:

| Metric | Without retraining | With final-refit retraining |
|---|---:|---:|
| RMSE | 915.73 | 893.48 |
| MAE | 699.52 | 693.35 |
| WMAPE | 11.34% | 11.24% |
| Bias | 122.12 | 98.23 |

Bias is calculated as mean prediction minus mean actual sales. Positive bias
therefore indicates systematic overprediction, while negative bias indicates
underprediction.

The experiment therefore shows that retraining adapts successfully to the
targeted promo-effect drift while highlighting why model promotion should be
evaluated across business-relevant segments rather than by a single aggregate
metric alone.

---

# 🧠 Business Context

The current project focuses on the technical MLOps lifecycle for sales forecasting.

In a real business environment, forecasts can be connected to operational decision layers such as:

- inventory planning
- staffing recommendations
- promotion planning
- demand risk alerts
- understock and overstock prevention
- store-level replenishment support
- capacity planning
- revenue forecasting

The forecasting model itself produces predicted demand, but the production value comes from turning those forecasts into operational decisions.

A possible extension would be a business decision layer that translates forecasts into actions such as:

| Forecast Signal | Possible Business Action |
|---|---|
| Expected high demand | Increase stock or staffing |
| Expected low demand | Reduce inventory exposure |
| Forecast above baseline | Investigate promotion or seasonal effect |
| Forecast below baseline | Trigger demand risk alert |
| Large forecast error | Flag store/date combination for review |

This keeps the project focused on MLOps while leaving room for a realistic business-facing extension.

---

# 🔁 When Retraining Happens

The controlled lifecycle demo uses persistent forecast-performance degradation
as its automatic retraining signal.

The monitoring architecture also produces additional signals that could be
connected to future retraining policies, including:

- feature drift
- data-quality failures
- scheduled retraining windows
- minimum newly available ground-truth volume
- explicit operational trigger flags

Once retraining is triggered, a new candidate is trained and compared with the
current champion on untouched chronological validation data. Only an accepted
candidate proceeds to final refitting and champion promotion.

This prevents uncontrolled model replacement and supports stable production
behavior.

---

# 🧪 End-to-End Lifecycle Demo

The repository includes demo scripts for simulating an operational forecasting lifecycle.

The demo can simulate:

- forecast batch generation
- newly available ground truth
- forecast performance evaluation
- retraining trigger checks
- performance comparison with and without retraining

After starting the local Docker Compose stack and running the initial training pipeline, execute:

```bash
make demo-forecasting-lifecycle
```

Each lifecycle run writes its monitoring history directly to the output file
specified for the simulation, for example:

```text
results/promo_mild_weights_with_retraining.csv
```

The Streamlit dashboard automatically discovers compatible lifecycle result
files under `results/`. Completed and partially running simulations can
therefore be inspected without copying or renaming result files.

The dashboard provides:

- operational monitoring for one selected lifecycle run
- rolling RMSE, MAE and bias
- drift ramp-up visualization
- retraining and final-refit promotion events
- interactive comparison of matching runs with and without retraining

Open the Streamlit monitoring dashboard at:

```text
http://localhost:8501
```

Because the local `results/` directory is mounted into the API container,
lifecycle output becomes available to the dashboard immediately.

---

# ☁️ Technology Stack

| Layer | Technologies |
|---|---|
| Modeling | Python 3.12, XGBoost, scikit-learn, Pandas |
| Serving | FastAPI, Streamlit |
| MLOps | MLflow, Prefect |
| Observability | Prometheus, Grafana |
| Runtime | Docker, PostgreSQL |
| Cloud and IaC | Google Cloud Run, Artifact Registry, Cloud Storage, Terraform |
| CI/CD | GitHub Actions |

---

# 📁 Project Structure

```text
.
├── configs/               # environment, training and monitoring configs
│   ├── dev.yaml
│   ├── staging.yaml
│   ├── prod.yaml
│   ├── gcp.yaml
│   ├── monitoring.yaml
│   └── training.yaml
│
├── src/                   # application source code
│   ├── api/               # FastAPI application and request schemas
│   ├── data/              # ingestion, validation, features and splits
│   ├── deployment/        # deployment CLI and cloud config
│   ├── inference/         # forecasting inference logic and model loading
│   ├── monitoring/        # drift, performance, serving and trigger logic
│   ├── training/          # training, evaluation and registration
│   └── utils/             # shared utilities
│
├── flows/                 # Prefect training and retraining flows
├── infrastructure/        # Terraform infrastructure
├── monitoring/            # Prometheus configuration
├── scripts/               # helper scripts and lifecycle demos
├── tests/                 # unit and integration tests
├── docs/                  # deployment docs and screenshots
└── .github/workflows/     # CI/CD pipeline
```

Key experiment and visualization files include:

```text
scripts/
├── run_model_backtest.py
├── tune_model.py
├── run_performance_demo.py
├── simulate_ground_truth.py
└── plot_retraining_comparison.py

docs/images/
├── streamlit_dashboard.png
├── streamlit_retraining_comparison.png
└── promo_mild_weights_comparison.png
```

---

# ⚡ Quick Start

The local demo starts the full MLOps stack with Docker Compose:

- FastAPI forecasting API
- Streamlit monitoring dashboard
- MLflow tracking server and model registry
- Prefect orchestration server
- PostgreSQL backend
- Prometheus metrics
- Grafana dashboard

After starting the services, run the training pipeline once to register an initial champion model.

## 1️⃣ Clone repository

```bash
git clone https://github.com/SL14-SL/mlops-sales-forecasting.git
cd sales-forecasting-mlops
```

---

## 2️⃣ Configure environment

```bash
cp .env.example .env
```

Set required variables:

* API_KEY
* GCP configuration, optional for local development

---

## 3️⃣ Start local services

```bash
make dev-up
```

The local services are available at:

| Service | Local URL |
|---|---|
| Forecasting API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Prefect | http://localhost:4221 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## 4️⃣ Run training pipeline

```bash
make train-force
```

This executes:

* ingestion
* validation
* feature engineering
* temporal splitting
* model training
* MLflow logging
* model registration

---

## 5️⃣ Run quality checks

```bash
make test
make lint
```

The test suite covers training, inference, feature engineering, monitoring and
lifecycle behavior. Ruff is used for static code-quality checks.

---

# 🔧 Configuration

The platform follows a configuration-driven architecture.

## Environment configs

* `configs/dev.yaml`
* `configs/prod.yaml`
* `configs/gcp.yaml`
* `configs/monitoring.yaml`
* `configs/training.yaml`

## Environment switching

Local Make targets run the containerized stack in development mode. The
development lifecycle commands explicitly set `APP_ENV=dev` and use the local
MLflow, Prefect and API service addresses.

Production helper targets set `APP_ENV=prod` and connect to the configured
cloud services. For example:

```bash
make train-force-prod
```

`APP_ENV` selects the environment-specific YAML configuration. Environment
variables such as API URLs, credentials and service endpoints remain
independent configuration inputs and are not automatically ignored merely
because development mode is active.

---

# ☁️ Deployment

Infrastructure is provisioned with Terraform.

Services can be deployed to Google Cloud using:

* Cloud Run
* Artifact Registry
* Google Cloud Storage
* GitHub Actions
* Workload Identity Federation

## Terraform

```bash
cd infrastructure
terraform init
terraform apply
```

## GitHub Actions

CI/CD automatically handles:

* linting
* testing
* API smoke testing
* Terraform validation and planning
* Docker image builds
* vulnerability scanning
* container registry publishing
* Cloud Run deployment

When the required GCP infrastructure, repository variables, secrets and
authentication are configured, pushes to `main` trigger the complete CI/CD
workflow for validation, image publication and Cloud Run deployment.

<p align="center">
  <img src="docs/images/ci_pipeline.png" width="100%">
</p>

<p align="center">
  <em>GitHub Actions CI/CD pipeline with linting, tests, API smoke checks, container builds, vulnerability scanning and cloud deployment steps.</em>
</p>

---

# 📈 API Endpoints

The forecasting API exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/docs` | Interactive Swagger documentation |
| `GET` | `/livez` | Process liveness check |
| `GET` | `/readyz` | Model and service readiness check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/monitoring/summary` | Monitoring status summary |
| `POST` | `/predict` | Single or batch forecast generation |
| `POST` | `/admin/reload-model` | Reload the active registry model |

For a deployed environment, prepend the active API base URL to the endpoint,
for example `https://YOUR_API_URL/predict`.

---

# 📦 Dataset

The project uses the Rossmann Store Sales dataset as a realistic store-level
sales forecasting use case.

The data contains daily observations for multiple stores together with
information about promotions, store availability, school holidays, state
holidays and store metadata. These fields support the temporal, promotional
and calendar-based feature engineering used by the forecasting pipeline.

The forecasting scenario is used to demonstrate:

- temporal feature engineering
- data validation
- time-aware splitting and walk-forward evaluation
- model training and hyperparameter tuning
- experiment tracking and model registry workflows
- API-based forecast serving
- prediction logging
- performance and drift monitoring
- automated retraining and controlled model promotion
- CI/CD deployment

The dataset and forecasting model provide a concrete application scenario, but
they are not the primary focus of the repository. The main focus is the
production-oriented MLOps infrastructure surrounding the model: reproducible
data processing, orchestration, model serving, monitoring, retraining, model
registry workflows and deployment automation.

Raw dataset files are intentionally excluded from version control.
---

# 🎯 Project Goals

This repository focuses on the operational layer of machine learning systems:

* production-oriented ML engineering
* model serving and API deployment
* experiment tracking and model registry workflows
* monitoring and observability
* reproducible training and inference pipelines
* CI/CD for ML services
* automated retraining and model promotion
* forecasting-specific inference handling
* cloud-native deployment patterns

The emphasis is on reliable ML infrastructure — not just model training or notebook experimentation.

---

# 🔁 Reusable Use Cases

The architecture can be adapted to other forecasting and regression problems
in which predictions must be served, monitored and improved over time.
Examples include demand, inventory, revenue, traffic, workload, staffing and
energy-consumption forecasting.

The reusable components are the data and training orchestration, experiment
tracking, registry-based promotion, API serving, monitoring, retraining and
deployment workflows. Forecasting-specific schemas, features and evaluation
policies can be replaced for a different application domain.

Raw dataset files are intentionally excludd from version control.

---

# ⚠️ Limitations

This repository is a production-oriented portfolio showcase, not a fully managed enterprise forecasting platform.

For a real enterprise deployment, I would additionally consider:

- centralized cloud logging and alerting
- stricter IAM scoping per environment
- managed secret rotation
- explicit SLO definitions
- external event calendars for local and business-specific events
- shadow model evaluation or canary deployment
- broader multi-horizon and multi-scenario backtesting
- richer forecast explainability
- hierarchical forecasting support
- probabilistic forecasts and prediction intervals
- cost monitoring and budget alerts
- data privacy controls for business-specific datasets
- blue/green deployment strategies
- a fully untouched final temporal test period after hyperparameter selection

The goal of this project is to demonstrate realistic MLOps architecture patterns in a compact and reproducible showcase.

---

# 📄 License

MIT License

---

# 👨‍💻 Author

**Steffen Lauterbach**  
MLOps Engineer

Focused on production-oriented ML systems, model deployment, monitoring, retraining workflows and cloud-native ML infrastructure.

LinkedIn:  
https://www.linkedin.com/in/92-steffen-lauterbach