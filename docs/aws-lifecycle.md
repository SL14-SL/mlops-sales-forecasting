# AWS model lifecycle and GCP parity

AWS now runs the same `run_prefect_model_lifecycle` and `run_auto_retraining`
services as the existing project. The earlier isolated training job remains
available for smoke tests, but is no longer the Terraform Fargate task's entry point.

## Functional mapping

| Capability | GCP / existing lifecycle | AWS lifecycle |
| --- | --- | --- |
| Training and evaluation | Project training pipeline | Same pipeline in ECS/Fargate |
| Tracking and model registry | Configured shared MLflow service | Configured shared MLflow service |
| Challenger assignment | MLflow `challenger` alias | Same implementation |
| Champion comparison | Champion evaluated on candidate validation split | Same implementation |
| Promotion guardrails | Overall, segment and bias checks | Same implementation |
| Release activation | Manifest and active pointer | Same implementation with S3 conditional writes |
| Serving model | Existing release/export mechanism | Portable registered model copied into S3 release |
| Publication failure | Restore previous champion alias | Same implementation |
| Running API update | Cloud Run deployment / admin reload | ECS rolling restart and expected-release readiness check |
| Monitoring and delayed labels | Existing monitoring services | Same services, reading S3 logs and ground truth |
| Conditional retraining | Monitoring policy, cooldown, duplicate detection | Same policy in scheduled/manual Fargate task |
| Model rollback | Existing release-pointer operations | Validated prior bundle, registry champion and pointer alignment, ECS refresh |
| API image rollback | Cloud Run workflow | Terraform-managed ECS image digest rollback workflow |
| Delivery authentication | GitHub OIDC | GitHub OIDC |
| API ingress | Cloud Run HTTPS | API Gateway HTTPS, internal ALB, ECS |

The two clouds share behavior; they do not require identical infrastructure.
The default API remains a single-instance demo. Existing Prometheus/Grafana
configuration remains usable; this change does not provision a managed metrics
stack on either cloud. Public AWS ingress still excludes admin and metrics routes.

## Managed MLflow option

This project can also provision MLflow on ECS/Fargate with RDS and S3. See
[Managed MLflow on AWS](aws-mlflow.md). In managed mode Terraform supplies the
endpoint and token ARN automatically; an existing external MLflow service is
not required. The external-service setup below still applies when managed mode
is disabled.

## Shared services

The existing GCP Terraform module accepts an externally configured MLflow URI;
it does not provision a durable registry server. AWS follows that same boundary.
Supply a reachable HTTPS MLflow tracking/registry service with a durable
database backend and a persistent artifact store. All training tasks in one
environment must use that same service and experiment. Do not use a laptop's
`localhost`, a task-local SQLite file, or a container-local artifact directory.

For MLflow artifacts, either use the server's artifact proxy or a persistent S3
artifact location. The task role can access `s3://<project-bucket>/mlflow/*`.
If the registry uses another artifact bucket, grant the training task access
there as well. The inference API only needs the portable release in the project
bucket; it does not need credentials to contact the registry.

Optional `PREFECT_API_URL` connects flows to an existing Prefect service.
An empty value runs an ephemeral local Prefect server for each task; runs are
not centrally visible in Prefect in that mode. MLflow tracking still persists.
Both services must be reachable through the task network, which permits HTTPS
outbound. For private endpoints, add the necessary network connectivity.

Optional Secrets Manager ARNs inject `MLFLOW_TRACKING_TOKEN` and
`PREFECT_API_KEY`. The ARN, not its value, enters Terraform state. These secrets
must exist before the task starts. Customer-managed KMS keys additionally need
`kms:Decrypt` grants for the execution role. Registry authentication should use
a token-capable HTTPS gateway when the chosen server does not accept tokens itself.

## GitHub configuration

Create one GitHub environment (`dev`, `staging`, or `prod`) per deployment.
Use the following environment variables:

| Variable | Required | Meaning |
| --- | --- | --- |
| `AWS_ACCOUNT_ID` | Yes | Target AWS account |
| `AWS_REGION` | Yes | Example: `eu-north-1` |
| `AWS_DEPLOY_ROLE_ARN` | Yes | Existing deployment role trusted through GitHub OIDC |
| `AWS_TF_STATE_BUCKET` | Yes | Existing, private, versioned S3 state bucket |
| `MLFLOW_TRACKING_URI` | Yes | Shared MLflow HTTPS URL |
| `PREFECT_API_URL` | No | Shared Prefect HTTPS API URL |
| `MLFLOW_TOKEN_SECRET_ARN` | No | Secret containing tracking token |
| `PREFECT_KEY_SECRET_ARN` | No | Secret containing Prefect API key |
| `AWS_API_IMAGE_DIGEST` | No | Explicit digest for plan-only runs |
| `AWS_TRAINING_IMAGE_DIGEST` | No | Explicit digest for plan-only runs |

Scheduled retraining is opt-in: set the **repository** variable
`AWS_RETRAINING_ENABLED=true` after the initial deployment. The daily schedule
runs at 05:00 UTC against `dev`; its policy decides whether training is warranted.
Other environments can run manually or receive their own configured schedule.

Restrict the OIDC role trust to this repository and its GitHub environment.
Read the actual subject prefix before constructing the trust condition:

```bash
gh api repos/SL14-SL/mlops-sales-forecasting/actions/oidc/customization/sub \
  --jq '.sub_claim_prefix'
```

Append `:environment:dev` to that prefix (adapt for each environment), and require
audience `sts.amazonaws.com`. GitHub may include immutable owner/repository IDs
in the prefix; do not assume the older name-only form. The role needs Terraform resource-management
permissions for this module's S3, ECR, IAM, ECS, EC2 networking, ELB, API Gateway,
CloudWatch and Secrets Manager resources. It also needs ECR push and ECS
RunTask/DescribeTasks/StopTask, with PassRole limited to the task and execution
roles. Its state permissions must cover Get/Put on the state object and
Get/Put/Delete on the `.tflock` object, plus bucket listing.

The account's OIDC provider, deployment role, state bucket, training data and
secret values are prerequisites outside this project module. An external shared
MLflow service is required only when managed MLflow is disabled.
They are not created by running the workflow. No cloud deployment was performed
while implementing this change.

## First deployment

1. Run **Deploy to AWS**, selecting the environment, `enable_serving=false`,
   and `apply_changes=true`. It creates ECR repositories, builds and pushes both
   images, then applies the complete module with the API stopped. The initial
   repository-only apply is intentional bootstrap work; the full plan follows.
2. Upload `train.csv` and `store.csv` to the `lifecycle_raw_data_uri` Terraform
   output, for example `s3://<bucket>/data/raw/`. New training batches go in
   `data/raw/new_batches/`; delayed labels go in the configured monitoring input
   location described by the existing monitoring runbook.
3. Set the API key value in the `api_key_secret_arn` output's Secrets Manager
   secret. Populate optional tracking/Prefect credential secrets too.
4. Run **AWS model lifecycle**, operation `train`. Leave `estimators` empty for
   the configured model size. A smoke value such as `20` is an explicit override;
   it does not bypass evaluation or promotion guardrails.
5. Confirm that MLflow contains the registered version and champion, and S3
   contains a complete release and `active_serving_release.json`.
6. Run **Deploy to AWS** again with `enable_serving=true`, `apply_changes=true`.
   ECS starts the API from S3. The workflow checks readiness and unauthenticated
   prediction rejection. Use the existing authenticated prediction smoke test
   to verify forecasts.

For a preview use `apply_changes=false`; this creates no repositories/images.
It reuses configured digests or current state outputs. For a completely empty
state it uses a zero SHA placeholder **only for planning**. Applying always builds
new images and resolves real ECR digests before producing the deployment plan.
Plans are printed in workflow logs; GitHub environment protection can control
who may execute applying workflows.

ECR image tags contain the commit, run ID and run attempt, and remain immutable.
Serving and lifecycle operations for the same environment share one GitHub
concurrency group. Avoid concurrent local training or manual task launches,
especially while a workflow is running. If a workflow is cancelled after ECS
launch, inspect/stop its task before starting another mutation.

## Subsequent training and retraining

The Terraform Fargate task now executes:

```text
python -m mlops_sales_forecasting.aws.lifecycle train
```

The lifecycle reads canonical S3 data, logs and registers the model, assigns
challenger, evaluates the current champion on the same validation set, applies
promotion guardrails, and publishes only a successful champion. Every model file
and serving auxiliary artifact is checksummed. A complete bundle is loaded and
validated before pointer activation. Serving loaders verify every declared
artifact before model deserialization.

If serving exists, the task forces an ECS rolling deployment, waits for its
completion, checks for circuit-breaker fallback and verifies `/readyz` names the
expected release. A failed update attempts rollback to the prior registered
release and refreshes ECS again. The task still fails so the incident stays
visible. For initial training without an API service, publication is sufficient;
the API is deployed in the next step.

`auto-retrain` refreshes monitoring signals and applies the same retraining policy
as GCP. The successful-decision marker is recorded only after the ECS refresh
hook succeeds, so a failed refresh is not marked as a successfully completed
retraining cycle. Supply actual delayed labels and incremental batches; a
schedule alone does not create new observations.

Run these operations with the **AWS model lifecycle** workflow, or after local
Terraform initialization with the AWS CLI authenticated:

```bash
python scripts/run_aws_task.py train
python scripts/run_aws_task.py auto-retrain
python scripts/run_aws_task.py refresh
python scripts/run_aws_task.py rollback-model
```

The launcher uses the deployed task definition, waits for termination, propagates
nonzero container exit codes, and stops a task exceeding its 40-minute deadline.
Detailed results and failure messages are in the existing CloudWatch task log.
Large training jobs can require a longer task/workflow deadline.

## Rollback

For the isolated 95-day static-versus-managed comparison on Fargate, see
[AWS lifecycle simulation](aws-simulation.md). This uses simulation dates for
cooldown evaluation and does not change the live API champion.

**Model:** `rollback-model` validates the previous portable bundle and checks its
registered version, then restores the `champion` alias, activates the old pointer
and refreshes ECS. If pointer activation fails, it restores the prior alias.
If ECS refresh fails after activation, the desired pointer and alias remain on
the rollback target; inspect the task and rerun `refresh`. Registry and S3 are
separate stores, so compensation is not a distributed transaction.

Only releases produced by the new registry-backed path qualify. A legacy
`aws-demo` smoke release has no shared registry identity and cannot be treated
as a registered rollback target. Bootstrap a new registry-backed champion first.

**Application image:** use **Roll back AWS API image**, providing a previously
tested API SHA-256 digest. Its default only plans; `apply_changes=true` applies
the Terraform-managed image change and checks readiness. Model release and
training image stay on their current values. Choose an API version compatible
with the active serving manifest and forecasting state.

## Local Terraform and state migration

Copy `terraform.tfvars.example` and fill real values and image digests. Initialize
against the same state key used by GitHub:

```bash
terraform -chdir=infrastructure/terraform-aws init \
  -backend-config="bucket=YOUR_STATE_BUCKET" \
  -backend-config="key=SL14-SL/mlops-sales-forecasting/dev/terraform.tfstate" \
  -backend-config="region=eu-north-1" \
  -backend-config=encrypt=true \
  -backend-config=use_lockfile=true
terraform -chdir=infrastructure/terraform-aws plan
```

If an existing local state manages live resources, back it up and use
`terraform init -migrate-state` with those backend settings. Do not initialize an
empty remote state over live resources or recreate them. The supplied archive
records that the earlier demo resources were destroyed and its state was empty.
The `environment=dev` default preserves the previous resource naming convention.

Disable serving with `api_serving_enabled=false`; avoid full destroy when only
stopping the API. This removes ingress resources as documented in AWS serving.

## Validation boundary

Local integration tests run real XGBoost training and a real MLflow database
through registration, promotion, non-promotion, publication failure recovery,
portable serving without the original artifact store, and model rollback.
Unit tests cover S3 adapters/pointers, AWS configuration, ECS circuit fallback,
readiness mismatch and refresh error handling. Workflow syntax is checked with
actionlint. Terraform formatting is checked and AWS validation is added to CI.

This implementation has not been exercised against a live AWS account or shared
remote registry. The new cloud workflow, IAM integration, endpoint reachability
and image build still need that cloud smoke test. Terraform provider startup in
the editing environment is blocked by its Unix-socket restriction, so local
`terraform validate` could not complete; GitHub CI must finish that check.

References: [MLflow server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/),
[MLflow artifact stores](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/),
[ECS service update](https://docs.aws.amazon.com/cli/latest/reference/ecs/update-service.html).
