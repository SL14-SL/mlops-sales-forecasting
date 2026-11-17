# AWS training

The normal AWS task now runs the full shared model lifecycle in ECS/Fargate:
training, evaluation, shared MLflow registration, challenger assignment, champion
comparison, gated promotion, portable S3 release activation, and ECS refresh.
The same task can execute monitoring-driven automatic retraining.

See [AWS lifecycle and GCP parity](aws-lifecycle.md) for setup, prerequisites,
manual and scheduled workflows, deployment, rollback and validation status.

The earlier isolated candidate-export entry points remain available:

- `mlops_sales_forecasting.aws.train`: isolated local/SageMaker-compatible job.
- `mlops_sales_forecasting.aws.fargate_train`: S3 download/export smoke adapter.

These two legacy commands intentionally do not register or promote candidates.
They are distinct from `mlops_sales_forecasting.aws.lifecycle`, the current
Terraform task entry point. SageMaker execution remains unverified; the full
lifecycle implementation targets ECS/Fargate.

The [historical smoke-test record](aws-training-smoke-history.md) documents the
previous cloud run and output artifacts. It does not establish verification of
the new shared-registry path.
