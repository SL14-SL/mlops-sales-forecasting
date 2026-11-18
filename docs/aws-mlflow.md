# Managed MLflow on AWS

The forecasting project can now provision its own shared MLflow service.
Set `mlflow_managed_enabled=true` locally, or the GitHub environment variable
`AWS_MANAGED_MLFLOW=true`. External MLflow remains supported when this flag is
false; the default preserves the previous Terraform behavior.

## Components

| Component | Purpose |
| --- | --- |
| ECS/Fargate MLflow task | MLflow 3.18.0 tracking and model registry |
| RDS PostgreSQL 16 | Persistent runs, metrics, model versions and aliases |
| Project S3 bucket, `mlflow/` | Persistent model and run artifacts |
| API Gateway HTTPS | Stable client endpoint without a custom domain |
| Lambda request authorizer | Bearer-token validation for every MLflow route |
| Internal ALB and VPC link | Restricted access from gateway to MLflow |
| Secrets Manager | MLflow token and RDS-managed master credentials |

The database is encrypted, not publicly accessible, and accepts connections
only from MLflow. The server connects using TLS (`sslmode=require`). The ALB
accepts traffic only from the gateway VPC link; the task accepts port 5000 only
from the ALB. The task has a public IP for outbound HTTPS, without a publicly
reachable application port. Gateway-to-ALB-to-container traffic uses HTTP inside
the VPC; this is not end-to-end TLS.

The RDS master password is generated and managed by RDS, never stored as a
Terraform variable or secret-version resource. MLflow reads it at startup via
its task role. The MLflow bearer token is initialized outside Terraform with
`scripts/ensure_mlflow_token.py`; an existing current token is retained. The
Lambda authorizer reads that secret and caches it for at most 60 seconds.
Tokens do not appear in Terraform outputs, task-definition environment variables
or command-line arguments. API Gateway authorization-result caching is disabled.

Training tasks receive the endpoint and token ARN automatically from Terraform.
No manual `MLFLOW_TRACKING_URI` or `MLFLOW_TOKEN_SECRET_ARN` GitHub variable is
needed in managed mode. The previous external-service variables apply only when
managed mode is disabled.

## Artifact transfers and the UI

The server uses `--no-serve-artifacts --default-artifact-root s3://<bucket>/mlflow`.
The SDK uploads/downloads artifacts directly using the caller's AWS role. This
avoids API Gateway's request-size limitation for large models. Metadata and
registry requests still use the authenticated HTTPS endpoint. The MLflow server
has read access to its S3 artifacts for browsing/presigned download links.

An MLflow token is not an AWS credential. A local SDK client requires both the
MLflow bearer token and AWS credentials with access to the artifact prefix.
The existing Fargate training role already has that S3 access.

For interactive UI access, authenticate your local AWS CLI session and run from
the project root after initializing its Terraform backend:

```bash
uv run --locked --extra aws python -m mlops_sales_forecasting.aws.mlflow_ui_proxy
```

Open `http://127.0.0.1:5001`. The loopback-only proxy retrieves the token from
Secrets Manager and adds it to upstream HTTPS requests, so you do not need to
paste a token into the browser. It rejects remote Host/Origin headers and
cross-site browser requests. Stop it with Ctrl+C. The raw gateway URL still
requires an Authorization header and is not an anonymously available UI.

This is a single shared project token, not multi-user identity or role-based
registry authorization. Every token holder can manage this MLflow deployment.

## Prerequisites

The GitHub OIDC provider and deployment role are already configured for the
project. Update that role's inline deployment policy with the supplied revised
`aws-dev-deploy-policy.json`: RDS, Lambda, three MLflow roles, required
service-linked roles and the MLflow token initializer require additional rights.
The policy's total compact size remains below IAM's role-inline-policy limit.
Validate it with IAM Access Analyzer in the target account before applying.
That AWS-side validation and a live deployment have not been executed here.

Keep the existing environment variables:

- `AWS_ACCOUNT_ID=713877988523`
- `AWS_REGION=eu-north-1`
- `AWS_TF_STATE_BUCKET=mlops-tfstate-713877988523-eu-north-1`
- `AWS_DEPLOY_ROLE_ARN=arn:aws:iam::713877988523:role/mlops-sales-forecasting-dev-github-deploy`

Enable managed MLflow:

```bash
gh variable set AWS_MANAGED_MLFLOW --env dev --body true
```

For local Terraform commands also add `mlflow_managed_enabled = true` to the
existing untracked `infrastructure/terraform-aws/terraform.tfvars`. Do not replace
its other settings or enter token/password values there.

The database instance default is `db.t4g.micro`, PostgreSQL major version 16,
20 GiB encrypted gp3 storage, and one Availability Zone. Verify availability
before the first apply if the account/region differs. RDS quotas, service-linked
role creation and account organization controls are checked only by the real
AWS plan/apply. The database is a billable resource while running.

## First deployment

After merging this patch and updating the role policy, run a plan-only workflow:

```bash
gh workflow run deploy-aws.yml --ref main \
  -f environment=dev -f enable_serving=false -f apply_changes=false
```

On an empty state, the plan uses placeholder image digests solely for preview.
The applying workflow first creates ECR repositories, builds/pushes API, training
and MLflow images, resolves real digests, and then plans/applies the full stack.
There is no fake registry URI and no job-local SQLite backend in managed mode.

When the plan is understood, run:

```bash
gh workflow run deploy-aws.yml --ref main \
  -f environment=dev -f enable_serving=false -f apply_changes=true
```

This starts MLflow and RDS while forecasting serving remains disabled. Database,
load-balancer and gateway provisioning can take a while. The workflow initializes
the MLflow token and performs real SDK operations: experiment, metric, artifact,
model registration, alias assignment, model download and prediction. It also
checks that missing and invalid bearer tokens are rejected.

No forecasting training is started by this deployment smoke test. The separate
smoke model has an infrastructure-only name and does not change the forecasting
model's champion/challenger aliases.

## Persistence check

Run **Verify AWS MLflow persistence** after successful provisioning:

```bash
gh workflow run verify-mlflow-aws.yml --ref main -f environment=dev
```

It creates a registered smoke model, forces an ECS replacement, verifies the
requested deployment actually completed with a different running task, and
loads the same existing alias/model to reproduce its prediction. After the
AWS `services_stable` waiter returns, it polls the exact requested deployment
and the running task list for up to another 150 seconds. ECS can report stable
service counts before its `rolloutState` becomes `COMPLETED`; a temporarily stale
status is not an immediate failure. A failed requested deployment is rejected,
and an old completed deployment or unchanged tasks cannot count as success.
The same rollout verification is used for forecasting API refreshes. The registry
metadata persists in RDS and the model files persist in S3.

The operation briefly interrupts MLflow because the deployment intentionally
runs a single instance and avoids parallel database schema migrations. Do not
run external/manual training while this verification is running. GitHub AWS
workflows share the existing environment concurrency group.

Then continue the [forecasting lifecycle](aws-lifecycle.md): upload Rossmann
inputs, train, verify promotion, initialize the separate forecasting API key,
enable serving, run predictions and test rollback.

## Costs and stopping

RDS, Fargate, ALB, public IPv4, Secrets Manager and storage incur usage charges.
API Gateway, Lambda and logs add usage-based costs. This is a persistent demo
service, not a zero-cost or highly available production deployment. Check the
AWS cost estimate and set billing alerts before leaving it running unattended;
activating advanced AWS account features removed the earlier hard spend limit.

Setting `mlflow_desired_count=0` and applying Terraform stops the MLflow task,
but **does not stop RDS or ALB charges**. The database has deletion protection
enabled, seven-day automatic backups, and a required final snapshot on deletion.
Do not toggle `mlflow_managed_enabled=false` as a simple stop switch: that plans
removal of the managed infrastructure and requires a valid replacement external
tracking URI.

For full teardown, first choose and verify a backup plan. With the current real
image digests and managed mode still selected, plan/apply
`mlflow_database_deletion_protection=false` and choose a unique
`mlflow_final_snapshot_identifier` beginning with `mlops-sales-forecasting-dev-mlflow-`.
Then review a full destroy plan before applying it. Terraform preserves a final
RDS snapshot, which continues to incur storage costs. Non-empty S3 buckets and
ECR repositories intentionally refuse deletion; decide explicitly what to back
up and retain. Never empty them automatically just to make destroy succeed.
Keep the separate Terraform-state bucket, which is not managed by this module.

A final snapshot is not automatically restored on a new build. Database restore
must be an explicit operation and paired with retained S3 artifacts. Database
snapshots alone do not contain model files.

## Operational limits

- Single MLflow instance, single-AZ RDS; no high-availability guarantee.
- MLflow uses its dedicated RDS master user. Production setups should introduce
  a restricted application database user and a separate migration user.
- PostgreSQL TLS is required but this demo does not configure `verify-full` CA
  and hostname verification.
- RDS-managed password rotation requires an MLflow task restart to reload the
  credential. Arrange that restart when rotating credentials.
- Gateway requests have a 30-second timeout; large artifacts bypass the gateway,
  but unusually slow metadata/model-registration requests may still time out.
- Preserve RDS and S3 together; test backup restoration before relying on it.

## Verification completed while editing

Python unit/integration tests, Ruff, dependency audit, actionlint and Terraform
formatting are run locally. New tests exercise credential handling, token
validation/rotation/failure, token initialization, restart verification and
loopback proxy access restrictions. The dependency audit retains the repository's
existing MLflow advisory exception; this change does not broaden it.

Terraform installs the AWS and archive providers, but this editing environment
blocks the Unix sockets required to start them. Thus full `terraform validate`,
AWS plan/apply and the new container image build remain GitHub/target-account
checks. No live AWS resources were created by the editing agent.

References: [MLflow server CLI](https://mlflow.org/docs/latest/api_reference/cli.html),
[HTTP API Lambda authorizers](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-lambda-authorizer.html),
[MLflow artifact stores](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/).
