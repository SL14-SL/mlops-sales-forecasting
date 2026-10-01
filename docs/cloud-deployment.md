# Google Cloud deployment

This document describes the initial Google Cloud bootstrap and the
subsequent GitHub Actions deployment of **Sales Forecasting MLOps**.

## Architecture

The deployment uses:

- Google Cloud Storage for Terraform remote state
- Workload Identity Federation for keyless GitHub authentication
- Artifact Registry for container images
- Secret Manager for the API key
- Cloud Run for the serving API
- a dedicated runtime service account
- GitHub Environments for deployment-specific secrets

No long-lived Google service-account key is stored in GitHub.

## Prerequisites

Install and configure:

- Google Cloud CLI
- Terraform
- Docker
- GitHub CLI

A Google Cloud project with billing enabled is required.

Authenticate locally:

```bash
gcloud auth login
gcloud auth application-default login
```

Select the project:

```bash
gcloud config set project GCP_PROJECT_ID
```

## Create the GitHub repository

Create the repository before enabling Workload Identity Federation,
because the identity provider is restricted to one exact repository.

From the generated project root:

```bash
gh repo create \
  mlops-sales-forecasting-next \
  --private \
  --source=. \
  --remote=origin \
  --push
```

The repository can be made public later without changing the workload
identity configuration.

## Bootstrap Terraform state and GitHub authentication

Create the local bootstrap configuration:

```bash
cp \
  infrastructure/terraform-bootstrap/terraform.tfvars.example \
  infrastructure/terraform-bootstrap/terraform.tfvars
```

Edit `infrastructure/terraform-bootstrap/terraform.tfvars`:

```hcl
gcp_project_id  = "your-gcp-project-id"
storage_location = "EU"

enable_github_actions = true

github_repository_owner = "your-github-owner"
github_repository_name  = "mlops-sales-forecasting-next"
```

Initialize the bootstrap module:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  init
```

Review the plan:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  plan
```

Apply it:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  apply
```

The bootstrap creates:

- the protected Terraform state bucket
- a GitHub Workload Identity Pool
- an OIDC provider restricted to this repository
- a GitHub deployment service account
- the required IAM bindings

Display the outputs:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  output
```

## Configure GitHub repository variables

Read the Bootstrap outputs:

```bash
TF_STATE_BUCKET="$(
  terraform \
    -chdir=infrastructure/terraform-bootstrap \
    output \
    -raw \
    terraform_state_bucket
)"

WIF_PROVIDER="$(
  terraform \
    -chdir=infrastructure/terraform-bootstrap \
    output \
    -raw \
    github_workload_identity_provider
)"

DEPLOY_SERVICE_ACCOUNT="$(
  terraform \
    -chdir=infrastructure/terraform-bootstrap \
    output \
    -raw \
    github_deployer_service_account
)"
```

Configure the required repository variables:

```bash
gh variable set \
  GCP_PROJECT_ID \
  --body "your-gcp-project-id"

gh variable set \
  GCP_REGION \
  --body "europe-west1"

gh variable set \
  ARTIFACT_REGISTRY_REPOSITORY \
  --body "mlops-sales-forecasting-images"

gh variable set \
  GCP_WORKLOAD_IDENTITY_PROVIDER \
  --body "$WIF_PROVIDER"

gh variable set \
  GCP_DEPLOY_SERVICE_ACCOUNT \
  --body "$DEPLOY_SERVICE_ACCOUNT"

gh variable set \
  TF_STATE_BUCKET \
  --body "$TF_STATE_BUCKET"

gh variable set \
  GCS_STORAGE_LOCATION \
  --body "EU"

gh variable set \
  ALLOW_UNAUTHENTICATED \
  --body "false"
```

Optional external service URLs can also be configured:

```bash
gh variable set \
  MLFLOW_TRACKING_URI \
  --body "https://mlflow.example.com"

gh variable set \
  PREFECT_API_URL \
  --body "https://prefect.example.com/api"
```

Leave them unset while these external services are not available.

An optional explicit Cloud Run service name can be configured:

```bash
gh variable set \
  CLOUD_RUN_SERVICE_NAME \
  --body "mlops-sales-forecasting-dev-api" \
  --env dev
```

If it is omitted, the rollback workflow derives the service name from
the GitHub repository name and deployment environment.

## Configure GitHub Environments

Create the deployment environments:

```bash
for environment in dev staging prod; do
  gh api \
    --method PUT \
    "repos/{owner}/{repo}/environments/${environment}"
done
```

Add the API key separately to each environment. The command prompts
for the secret value without writing it to the repository:

```bash
gh secret set API_KEY --env dev
gh secret set API_KEY --env staging
gh secret set API_KEY --env prod
```

Use different keys for the three environments.

For `prod`, configure required reviewers through:

```text
GitHub repository
→ Settings
→ Environments
→ prod
→ Required reviewers
```

## Plan the deployment

The first workflow run prepares the foundational infrastructure,
publishes an immutable container image and creates a Cloud Run
deployment plan. It does not update the live Cloud Run service.

Start a development plan:

```bash
gh workflow run \
  deploy.yml \
  --field environment=dev \
  --field apply_changes=false
```

Watch the workflow:

```bash
gh run watch
```

The workflow uploads the human-readable Terraform plan as a workflow
artifact. Find the completed run:

```bash
gh run list \
  --workflow deploy.yml \
  --limit 5
```

Download the plan using its run ID:

```bash
gh run download \
  RUN_ID \
  --pattern "terraform-plan-dev-*"
```

Review `deployment-plan.txt` before requesting the apply.

The planning run may create or update foundational resources such as
Artifact Registry, the artifact bucket, service accounts and Secret
Manager. It does not change the live Cloud Run revision.

## Apply the deployment

After reviewing the plan, start a second workflow run:

```bash
gh workflow run \
  deploy.yml \
  --field environment=dev \
  --field apply_changes=true
```

The apply run:

1. refreshes the foundational infrastructure;
2. builds and publishes an immutable container image;
3. creates a fresh Terraform deployment plan;
4. applies that exact plan;
5. verifies that Cloud Run reports a ready service.

Watch its progress:

```bash
gh run watch
```

For `prod`, configure required reviewers in the GitHub `prod`
environment. The apply workflow cannot begin before approval.

The deployment summary contains the image URI, service name and
Cloud Run URI.


## Verify the deployment

Obtain the Cloud Run URI:

```bash
SERVICE_URI="$(
  gcloud run services describe \
    "mlops-sales-forecasting-dev-api" \
    --region "europe-west1" \
    --format "value(status.url)"
)"
```

For an IAM-protected service:

```bash
IDENTITY_TOKEN="$(
  gcloud auth print-identity-token
)"

curl \
  --include \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/livez"
```

Expected response:

```text
HTTP/2 200
```

Readiness can return `503` until an active serving bundle exists:

```bash
curl \
  --include \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  "${SERVICE_URI}/readyz"
```

A prediction additionally requires the application API key:

```bash
read -r -s -p "API key: " API_KEY_VALUE
echo

curl \
  --include \
  --request POST \
  --header "Authorization: Bearer ${IDENTITY_TOKEN}" \
  --header "X-API-Key: ${API_KEY_VALUE}" \
  --header "Content-Type: application/json" \
  --data '{"inputs":[{"Store":1,"Date":"2026-09-28","Open":1,"Promo":1,"StateHoliday":"0","SchoolHoliday":0}]}' \
  "${SERVICE_URI}/predict"

unset API_KEY_VALUE
```

The example date must be covered by the active release's known calendar.
Adjust the payload when the deployed release uses a different calendar range.

## Public access

The secure default is:

```text
ALLOW_UNAUTHENTICATED=false
```

This requires both Cloud Run IAM authentication and the application API
key where applicable.

For a public demonstration API, set:

```bash
gh variable set \
  ALLOW_UNAUTHENTICATED \
  --body "true"
```

The application API key still protects the prediction endpoint.

Do not enable public access for sensitive customer workloads without a
security review.

## Updating the service

Every manually started deployment builds an immutable image tagged with
the Git commit SHA.

After merging changes, first create and review a new deployment plan:

```bash
gh workflow run \
  deploy.yml \
  --field environment=dev \
  --field apply_changes=false
```

After reviewing the plan, apply the update:

```bash
gh workflow run \
  deploy.yml \
  --field environment=dev \
  --field apply_changes=true
```

Terraform updates Cloud Run to the immutable image produced by the
apply run while preserving the existing infrastructure.


## Rolling back Cloud Run

The Cloud Run rollback changes application traffic without modifying
the active model serving release.

Roll back automatically to the second-newest revision:

```bash
gh workflow run \
  rollback.yml \
  --field environment=dev
```

Select a specific revision:

```bash
gcloud run revisions list \
  --service "mlops-sales-forecasting-dev-api" \
  --region "europe-west1"
```

```bash
gh workflow run \
  rollback.yml \
  --field environment=dev \
  --field revision=REVISION_NAME
```

Watch the workflow:

```bash
gh run watch
```

For production, GitHub Environment reviewers should approve the
rollback before the job starts.

A Cloud Run rollback and a model rollback are separate operations:

- use this workflow for broken application or container revisions;
- use the serving-release rollback for a faulty model release.


## Destroying an environment

Initialize Terraform against the existing remote state before destroying
resources:

```bash
terraform \
  -chdir=infrastructure/terraform \
  init \
  -reconfigure \
  -backend-config="bucket=${TF_STATE_BUCKET}" \
  -backend-config="prefix=mlops-sales-forecasting-next/dev"
```

Review the destroy plan carefully:

```bash
terraform \
  -chdir=infrastructure/terraform \
  plan \
  -destroy \
  -var="gcp_project_id=your-gcp-project-id" \
  -var="environment=dev" \
  -var="container_image=unused" \
  -var="deploy_cloud_run=true"
```

Only after reviewing it:

```bash
terraform \
  -chdir=infrastructure/terraform \
  destroy \
  -var="gcp_project_id=your-gcp-project-id" \
  -var="environment=dev" \
  -var="container_image=unused" \
  -var="deploy_cloud_run=true"
```

The bootstrap state bucket is protected with `prevent_destroy` and must
not be deleted while any environment still uses it.