# AWS serving

The forecasting API has been deployed and tested on ECS/Fargate.
It loads a portable serving release from S3 and exposes selected routes
through an API Gateway HTTPS endpoint.

## Architecture

Requests follow this path:

1. API Gateway terminates public HTTPS.
2. A VPC link forwards requests to an internal Application Load Balancer.
3. The load balancer forwards requests to the Fargate API container.
4. The API loads its active serving bundle from S3.

Traffic between API Gateway, the internal load balancer, and the container
uses HTTP within the VPC. This is not end-to-end TLS.

Security groups restrict the load balancer to the VPC link and the API
container to the load balancer. The task uses a public IP for outbound AWS
service access; its application port is not publicly accessible.

## Serving release

A complete release contains:

- An MLflow-format XGBoost model with its input signature.
- Validated store metadata.
- Dated forecasting state.
- Known calendar features.
- A manifest containing artifact checksums.

`prepare_local_serving_release` in
`src/mlops_sales_forecasting/aws/serving_release.py` creates and validates
a local release without activating it.

The existing cloud exporter can publish the validated release to S3.
The API reads the active release pointer from the configured models path.

The initial cloud training export lacked store metadata. For the serving
smoke test, this file was reconstructed from the corresponding S3 input
`store.csv` using the project's validation function. Future training exports
include it directly.

No shared MLflow registry registration or automatic champion promotion
is implemented in this AWS candidate-to-serving path.

## API image

Build the AWS variant:

```bash
docker build \
  --build-arg INSTALL_AWS=true \
  -t mlops-sales-forecasting-api:aws-local \
  .
```

Without the build argument, the existing image build remains available.

The API container runs as a non-root user. Its ECS task definition uses
1 vCPU and 4 GB memory.

## Authentication and routes

The API key is generated outside Terraform and stored in Secrets Manager.
ECS injects it at container startup using the execution role.

The application task role can read serving artifacts and write inference
monitoring files. It cannot activate or overwrite serving releases.

The public gateway exposes:

| Route | Authentication |
| --- | --- |
| `GET /livez` | Public |
| `GET /readyz` | Public |
| `GET /docs` | Public |
| `GET /openapi.json` | Public |
| `POST /predict` | Application-enforced `X-API-KEY` |

Admin and monitoring endpoints are not exposed through gateway routes.
API Gateway itself does not validate the application API key.

Secrets are not stored in Terraform variables or managed secret-version
resources. Do not print the key, commit it, or include it in screenshots.

## Verified smoke test

Verified on 2026-11-16:

- The complete serving bundle loads from S3.
- All published artifact checksums were verified.
- `/livez` returns a successful response.
- `/readyz` reports the expected active release.
- `/predict` without an API key returns HTTP 401.
- An authenticated HTTPS request returns predictions from the AWS service.
- Store 1, open on 2015-04-28 with promotion, returns 2524.30.
- Store 2, closed for the same date, returns 0.00.
- Prediction does not change the forecasting state.

The model uses only 20 trees. These checks establish serving functionality,
not production forecast quality.

## Endpoint and health checks

Read the deployed endpoint:

```bash
API_ENDPOINT=$(terraform -chdir=infrastructure/terraform-aws output -raw api_endpoint)
```

Check health:

```bash
curl --fail-with-body "$API_ENDPOINT/livez"
curl --fail-with-body "$API_ENDPOINT/readyz"
```

The Swagger interface is available at `/docs`.

## Operations and costs

The ECS service maintains the configured number of API tasks and replaces
unhealthy containers. Container and load-balancer health checks use `/readyz`.

The development setup uses one API task and is not a highly available
multi-instance deployment.

Container logs and API Gateway access logs have 14-day retention.
Gateway logs do not intentionally include request bodies or API keys.

The API container, public IPv4 address, and load balancer incur ongoing
usage while provisioned. S3, ECR, Secrets Manager, logs, and API requests
can also consume credits or incur charges.

Setting `api_desired_count = 0` and applying Terraform stops the API tasks,
but does not remove the load balancer or its ongoing charges.

A complete demo teardown must also remove the serving ingress resources.
Do not run a full Terraform destroy merely to stop serving: the same
Terraform configuration also manages training infrastructure and storage.