
# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2026-11-19

### Added
- AWS deployment support alongside the existing Google Cloud deployment.
- Verified AWS Fargate training execution and optional Amazon SageMaker training.
- Forecasting API deployment on AWS with HTTPS.
- Managed MLflow tracking infrastructure on AWS.
- AWS model lifecycle functionality, including model registration, evaluation, promotion, and serving release publication.
- AWS lifecycle simulation on ECS Fargate.
- AWS deployment, rollback, lifecycle, and MLflow verification workflows.
- AWS verification documentation, screenshots, and regression coverage.
- Optional AWS serving infrastructure and documented demo cleanup procedures.

### Changed
- Improved training container configuration and lifecycle log forwarding.
- Updated forecasting case studies and deployment documentation.
- Improved forecasting state contracts, promotional history consistency, and release handling.

### Fixed
- ECS deployment rollout verification.
- AWS training container project-root detection.
- MLflow task memory allocation.
- Concurrency safety for serving release pointer activation.
- Forecasting request ordering and promotional history consistency.
- Cloud Run deployment, readiness, release loading, and rollback behavior.
- Dependency security vulnerabilities.

### Security
- Updated vulnerable Python dependencies.
- Updated Docker base images and infrastructure dependencies.

## [1.0.1]

Previous published release.

## [1.0.0]

Initial published release.
