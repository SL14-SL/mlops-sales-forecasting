# Google Cloud Production Demo

## Purpose and Scope

This guide defines a reproducible production-oriented demonstration of the
Sales Forecasting MLOps serving layer on Google Cloud.

The demo proves:

- keyless GitHub Actions authentication through Workload Identity Federation;
- Terraform-managed cloud infrastructure;
- immutable container publication to Artifact Registry;
- reviewed plan-before-apply deployment;
- Cloud Run application health;
- Secret Manager integration;
- separation of application deployment and model-release activation;
- authenticated forecast serving when an active release is available;
- independent application and model rollback paths.

The provided Terraform deployment creates the serving infrastructure. It does
not provision managed MLflow or Prefect services.

A complete ready-state demonstration additionally requires:

- a reachable MLflow tracking and registry endpoint;
- Rossmann artifacts in the configured Cloud Storage bucket;
- an active serving-release pointer;
- a registered champion model accessible to the Cloud Run service.

## Prerequisites

Complete the infrastructure bootstrap and repository configuration described
in [Cloud deployment](cloud-deployment.md).

Before running this demo, verify that:

- Google Cloud billing is enabled;
- Workload Identity Federation is configured;
- required GitHub variables and environment secrets exist;
- the deployment environment has the intended approval rules;
- an external MLflow endpoint is configured when model serving is required.

## Plan Before Apply

Create a development deployment plan:

```bash
gh workflow run \
  deploy.yml \
  --field environment=dev \
  --field apply_changes=false

gh run watch
```

Find the completed run and download its plan:

```bash
gh run list \
  --workflow deploy.yml \
  --limit 5

gh run download \
  RUN_ID \
  --pattern "terraform-plan-dev-*"
```

Review `deployment-plan.txt` before requesting an apply. The plan run publishes
an immutable image and prepares infrastructure without updating the live Cloud
Run service.

## Apply the Deployment

After reviewing the plan, start the apply run:

```bash
gh workflow run \
  deploy.yml \
  --field environment=dev \
  --field apply_changes=true

gh run watch
```

The workflow:

1. prepares foundational infrastructure;
2. stores the API key in Secret Manager;
3. builds an image tagged with the Git commit SHA;
4. pushes the image to Artifact Registry;
5. creates a fresh deployment plan;
6. applies that exact plan;
7. verifies the Cloud Run service state.

## Stage 1: Verify Application Deployment

Resolve the service URI and obtain an identity token:

```bash
SERVICE_URI="$(
  gcloud run services describe \
    "mlops-sales-forecasting-dev-api" \
    --region "europe-west1" \
    --format "value(status.url)"
)"

IDENTITY_TOKEN="$(
  gcloud auth print-identity-token
)"
```

Verify liveness:

```bash
curl \
  --include \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/livez"
```

Expected result:

```text
HTTP/2 200
```

This proves that the deployed application revision is reachable. It does not
yet prove that a model can serve predictions.

## Stage 2: Verify Serving Readiness

Check readiness:

```bash
curl \
  --include \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/readyz"
```

HTTP `503` is expected until all of the following exist:

- a reachable MLflow registry;
- a registered champion model;
- a complete serving release in the configured artifact bucket;
- a valid active-release pointer;
- runtime access to every referenced artifact.

HTTP `200` proves that Cloud Run loaded a complete validated serving bundle.

## Stage 3: Verify a Forecast

Read the environment-specific API key without printing it:

```bash
read -r -s -p "API key: " API_KEY_VALUE
echo
```

Send a valid Rossmann request using a date covered by the active release's
known calendar:

```bash
curl \
  --include \
  --request POST \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  --header "X-API-Key: ${API_KEY_VALUE}" \
  --header "Content-Type: application/json" \
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
  "${SERVICE_URI}/predict"

unset API_KEY_VALUE
```

Replace the example date when it is not covered by the deployed known
calendar.

A successful response proves the complete path from authenticated Cloud Run
request through serving-release loading and forecasting inference.

## Evidence to Capture

For a portfolio or acceptance demonstration, retain:

- successful CI and Security workflow runs;
- reviewed Terraform plan;
- deployment workflow summary;
- immutable Artifact Registry image URI;
- Terraform outputs;
- Cloud Run revision and service URI;
- `/livez` response;
- `/readyz` response with active release ID;
- one successful authenticated prediction;
- MLflow model version and champion alias;
- active serving-release manifest.

Do not store API keys, identity tokens or other credentials in screenshots or
logs.

## Rollback Boundaries

Roll back a broken application revision:

```bash
gh workflow run \
  rollback.yml \
  --field environment=dev

gh run watch
```

A Cloud Run rollback changes application traffic without modifying the active
model release.

Use serving-release rollback when the application is healthy but the active
model bundle is faulty. The two rollback mechanisms are intentionally
independent.

## Demo Limitations

This is a production-oriented reference deployment, not a complete managed
platform offering.

The Terraform stack does not provision:

- managed MLflow;
- managed Prefect;
- a cloud training worker;
- organization-specific networking;
- centralized billing exports.

Those services must be provided separately when the complete training and
automated-retraining lifecycle is required in Google Cloud.

## Related Documentation

- [Cloud deployment](cloud-deployment.md)
- [System architecture](architecture.md)
- [Serving releases](serving-releases.md)
- [Operations runbook](operations-runbook.md)
