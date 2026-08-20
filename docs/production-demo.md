# Google Cloud Production Demo

## Purpose and Scope

This guide describes the temporary Google Cloud deployment used to demonstrate
the production lifecycle. It proves that the same training, release and
verification logic works outside the local Docker Compose environment.

It is intentionally cost-conscious and is not presented as a continuously
operated enterprise platform.

## Cloud Resources

Terraform provisions:

- Artifact Registry;
- a GCS artifact bucket;
- an MLflow Cloud Run service;
- a forecasting API Cloud Run service;
- service accounts and IAM bindings;
- Workload Identity Federation for GitHub Actions.

<p align="center">
  <img
    src="images/gcp_cloud_run_overview.png"
    width="70%"
    alt="MLflow and forecasting API services on Google Cloud Run"
  >
</p>

<p align="center">
  <em>
    Cloud Run services hosting the MLflow tracking server and the
    production forecasting API in europe-west1.
  </em>
</p>

The real prediction service is named `forecasting-api`. Avoid creating a second
placeholder service with a different name, because it causes infrastructure
drift and confusing public URLs.

## Required Local Configuration

Load the project environment after opening a new terminal:

```bash
set -a
source .env
set +a
```

At minimum, configure:

```text
GCP_PROJECT_ID=your-project-id
GCP_REGION=europe-west1
GCP_BUCKET_NAME=your-unique-artifact-bucket
MLFLOW_URL=https://your-mlflow-service.run.app
PREDICTION_API_URL=https://your-api-service.run.app/predict
PRODUCTION_API_URL=https://your-api-service.run.app
```

Do not commit production API keys or service credentials.

Authenticate when necessary:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
```

## Provision Infrastructure

```bash
terraform -chdir=infrastructure init
terraform -chdir=infrastructure fmt -check
terraform -chdir=infrastructure validate
terraform -chdir=infrastructure plan
terraform -chdir=infrastructure apply
```

Review every replacement or deletion in the plan before applying it. In
particular, verify Cloud Run service names, IAM targets and bucket operations.

Do not commit `tfplan`, `tfplan.txt`, `.terraform/` or state files.

## Upload Raw Demo Data

Raw data is excluded from Git. Upload it after the bucket exists:

```bash
make upload-raw-prod
```

Confirm the uploaded objects:

```bash
gcloud storage ls \
  "gs://${GCP_BUCKET_NAME}/data/raw/"
```

## CI/CD Deployment

GitHub Actions performs:

1. linting and tests;
2. API smoke testing;
3. API and MLflow container builds;
4. Trivy vulnerability scans;
5. Artifact Registry publication;
6. Cloud Run deployment;
7. production verification where configured.

<p align="center">
  <img
    src="images/ci_pipeline.png"
    width="100%"
    alt="GitHub Actions CI and deployment pipeline"
  >
</p>

<p align="center">
  <em>
    Successful GitHub Actions pipeline covering linting, tests,
    API smoke testing, vulnerability-scanned image builds and
    Cloud Run deployment.
  </em>
</p>

Required repository configuration includes the Artifact Registry path, project
and region variables, API secrets and Workload Identity Federation values.

## Bootstrap Production

The first production model requires a bootstrap run:

```bash
make train-bootstrap-prod
```

The target prepares the temporary MLflow demonstration instance before running
the production training pipeline. It should not be used when a champion already
exists in the current MLflow backend.

Successful output includes:

- candidate and final-refit run IDs;
- registered model version;
- immutable release ID;
- `champion_promoted: true`;
- `deployment_status: verified`;
- successful prediction-probe evidence.

## Verify Production Independently

```bash
make verify-prod
```

The verification script checks:

- the environment does not point to local services;
- API liveness;
- API readiness;
- expected release, model version and run ID;
- semantic prediction-probe execution;
- finite and valid prediction output.

<p align="center">
  <img
    src="images/terminal_output_make_verify_prod.png"
    width="80%"
    alt="Successful production deployment verification"
  >
</p>

<p align="center">
  <em>
    Independent verification of the production API, active serving
    release, model lineage and semantic prediction behavior.
  </em>
</p>

Verification must use the API base URL for health endpoints and the `/predict`
URL only for prediction traffic.

## Inspect Cloud Run

```bash
gcloud run services describe forecasting-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(metadata.name,status.url,status.traffic)'

gcloud run services describe mlflow-server \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(metadata.name,status.url,status.traffic)'
```

Inspect errors:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND severity>=ERROR' \
  --project "$GCP_PROJECT_ID" \
  --freshness=30m \
  --limit=100
```

## Cost-Conscious MLflow Configuration

The portfolio demonstration uses:

- one MLflow application worker;
- one warm Cloud Run instance while the demo is active;
- SQLite under `/tmp` as the tracking backend;
- GCS as artifact and serving-release storage.

The single-instance configuration avoids requests reaching different SQLite
databases concurrently. It does not make `/tmp` durable. Revision replacement,
instance replacement or scale-to-zero can remove the tracking database.

For continuous production operation, use PostgreSQL or Cloud SQL and configure
the MLflow backend URI accordingly.

## Memory and Scaling

MLflow UI and registry operations can exceed a 2 GiB Cloud Run limit. The
Terraform configuration should be the source of truth for memory and scaling;
manual `gcloud run services update` changes otherwise create Terraform drift.

After emergency manual changes, update Terraform and apply it so the declared
and actual infrastructure match.

## Teardown

The demo should be removed when it is no longer required:

```bash
terraform -chdir=infrastructure plan -destroy
terraform -chdir=infrastructure destroy
```

Review retained buckets, Artifact Registry images and externally managed
resources separately. Destruction of cloud resources and data is irreversible.

## Related Documentation

- [Architecture](architecture.md)
- [Serving releases](serving-releases.md)
- [Monitoring and SLOs](monitoring-and-slos.md)

