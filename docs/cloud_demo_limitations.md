# Cloud Demo Limitations

## Purpose

The GCP deployment is designed as an ephemeral demonstration environment for
portfolio reviews, technical evaluations, and freelance project discussions.

It demonstrates:

- Infrastructure provisioning with Terraform
- Workload Identity Federation for GitHub Actions
- Container image build, scanning, and deployment
- Cloud Run deployment
- GCS-backed model and serving artifacts
- Model registration and serving-release lineage
- Post-deployment verification
- Monitoring, alerting, and rollback mechanisms

## MLflow backend

The temporary Cloud Run demonstration uses a local SQLite MLflow backend.

This backend is not durable across Cloud Run instance replacement and must not
be considered suitable for a continuously operated production environment.

Model artifacts and serving-release artifacts remain persisted in GCS, but
MLflow tracking metadata, registry versions, aliases, and experiment history
may be lost when the MLflow instance is replaced.

## Production recommendation

A continuously operated production deployment should use a persistent remote
MLflow backend, such as:

- Cloud SQL for PostgreSQL
- A managed PostgreSQL service
- Another supported durable relational database

Database credentials should be stored in Secret Manager and injected into the
MLflow service at runtime.

## Cost decision

Cloud SQL is intentionally omitted from the demonstration environment to avoid
continuous infrastructure costs while the platform is not actively being
demonstrated.

The local Docker Compose environment uses PostgreSQL and demonstrates the
persistent backend architecture without requiring a continuously running cloud
database.

## Demo lifecycle

The intended cloud-demo lifecycle is:

1. Provision infrastructure with Terraform.
2. Upload the required raw data.
3. Deploy MLflow and the prediction API.
4. Bootstrap a Champion and publish a serving release.
5. Run post-deployment verification.
6. Demonstrate monitoring and rollback.
7. Destroy the temporary cloud infrastructure after the demonstration.

This environment is ephemeral by design.