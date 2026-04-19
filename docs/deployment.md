# 🚀 Deployment Guide

This document describes how to deploy the MLOps system in a cloud environment.

The system is designed to be reproducible, containerized, and production-ready.

👉 This guide covers both local (Docker-based) and cloud (Terraform-based) deployment options.

---

## 🧠 Overview

The deployment includes:

- Prediction API (FastAPI)
- MLflow tracking & model registry
- Prefect orchestration
- Monitoring components
- Supporting infrastructure (storage, networking)

Deployment is handled via:

- Docker (services)
- Terraform (infrastructure)
- CI/CD (optional)

## ⚠️ Important

This system requires input data to run the training pipeline.

Before running the initial training step, make sure to:

- provide raw data in data/raw/
- ensure it matches the expected schema (see README)

👉 Deployment sets up the infrastructure — it does not include data.

---

## 📦 Requirements

Before deploying, make sure you have:

- Docker installed
- Terraform installed
- Access to a cloud provider (e.g. GCP for this example)
- Python environment for local testing

---

## ⚙️ Infrastructure Setup (Terraform)

Navigate to the infrastructure directory:

```bash
cd infrastructure/
```
Create a terraform.tfvars file based on the example:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Initialize Terraform:

```bash
terraform init
```

Preview the deployment:

```bash
terraform plan
```

Apply the infrastructure:

```bash
terraform apply
```

This will provision:

- compute resources (e.g. VM or container runtime)
- networking
- storage (for data + artifacts)

👉 Note: Configure your Terraform backend (e.g. remote state) and variables  
before applying in a real cloud environment.

---

## 🐳 Service Deployment (Docker)

Build the services:

```bash
docker-compose build
```

Start all services:

```bash
docker-compose up -d
```

This will launch:

- FastAPI prediction service
- MLflow tracking server
- Prefect orchestration service

---

## 🧠 Initial model setup

After the first deployment, no model is available yet.

You need to run an initial training step to bootstrap the system:

```bash
make train-force
```

This will:

- train the first model  
- register it in MLflow  
- set it as the initial champion  

👉 Without this step, the API cannot serve predictions.

👉 In production setups, this step is typically automated as part of a pipeline.

👉 This step depends on input data.

The training pipeline will automatically:

- ingest raw data from data/raw/
- validate and transform it
- generate features and training splits

👉 If no data is available, the pipeline will fail with a clear error message.

---

## 🔁 Pipelines

Once deployed, you can trigger pipelines:

### Training pipeline

```bash
python -m flows.training_flow
```

### Monitoring / simulation

```bash
python scripts/run_performance_demo.py
```

---

## 📊 Monitoring

After deployment, the system provides:

- model performance tracking
- drift detection
- alerting signals
- retraining triggers

You can access the services locally via:

- MLflow UI → http://localhost:5000  
- Prefect UI → http://localhost:4200  
- API → http://localhost:8000  
- Dashboard → http://localhost:8501  

---

## 🔐 Configuration

Environment variables and configuration files control:

- model parameters
- thresholds (drift, performance)
- infrastructure settings

A `.env.example` file is provided for local setup.

Copy it with:

```bash
cp .env.example .env
```

Adjust values depending on your environment.

---

## 🔐 GitHub Actions configuration

This project expects a small set of GitHub Actions secrets and variables  
for CI/CD and cloud deployment.

### Required GitHub Secrets

- GCP_WIF_PROVIDER  
- GCP_SA_EMAIL  
- API_KEY  

### Required GitHub Variables

- GCP_REGION  
- GCP_ARTIFACT_REPO  
- MLFLOW_URL  
- GCP_PROJECT_ID  
- GCP_BUCKET_NAME  

### Security and authentication

- Workload Identity Federation (OIDC) is used to authenticate with GCP  
- no long-lived cloud credentials are stored in GitHub  
- only minimal application-level secrets are required  

👉 This setup reduces security risks and keeps infrastructure access tightly controlled.

---

### How to obtain values

Most required variables can be retrieved from your infrastructure setup.

For example:

- Terraform outputs (e.g. project ID, bucket names, regions)  
- Cloud provider configuration (e.g. service accounts)  
- Existing MLflow or API endpoints  

👉 In a real setup, these values are typically managed via infrastructure-as-code  
and injected into CI/CD pipelines.

---

## 🚀 CI/CD (optional)

You can automate deployment via CI/CD:

- build Docker images  
- run tests  
- apply Terraform  
- deploy updated services  
- container images are built and scanned for vulnerabilities before deployment  
- deployments only proceed after successful build and validation

Typical flow:

Commit → CI pipeline → Build → Scan → Deploy → Monitor

---

## 🧭 Deployment Philosophy

This system follows key MLOps principles:

- reproducibility (versioned data + models)  
- observability (monitoring + alerts)  
- automation (pipelines + retraining)  
- safety (only better models are deployed)  
- secure-by-default deployment via automated CI/CD and vulnerability scanning

👉 The goal is not just to deploy models, but to keep them reliable over time.

---

## 📩 Questions?

If you want help deploying this system in your environment:

→ feel free to reach out