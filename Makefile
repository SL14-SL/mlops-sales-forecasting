.DEFAULT_GOAL := help

UV ?= uv
COMPOSE ?= docker compose


.PHONY: help
help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)


.PHONY: sync
sync: ## Synchronize Python dependencies
	$(UV) sync


.PHONY: lint
lint: ## Run Ruff linting
	$(UV) run ruff check .


.PHONY: test
test: ## Run the complete test suite
	$(UV) run pytest


.PHONY: check
check: lint test ## Run all required local quality checks


.PHONY: api-config
api-config: ## Validate and display the Compose configuration
	$(COMPOSE) config


.PHONY: api-build
api-build: ## Build the API container image
	$(COMPOSE) build api


.PHONY: api-up
api-up: ## Start the API using the existing image
	$(COMPOSE) up --detach api


.PHONY: api-rebuild
api-rebuild: ## Rebuild the image and recreate the API container
	$(COMPOSE) up --detach --build --force-recreate api


.PHONY: api-ps
api-ps: ## Show the API container status
	$(COMPOSE) ps api


.PHONY: api-logs
api-logs: ## Follow API container logs
	$(COMPOSE) logs --follow api


.PHONY: api-down
api-down: ## Stop and remove the local Compose stack
	$(COMPOSE) down


.PHONY: api-restart
api-restart: api-down api-up ## Rebuild and restart the API

.PHONY: prefect-up
prefect-up: ## Build and start the local Prefect server
	$(COMPOSE) --profile orchestration up --detach --build prefect-server


.PHONY: prefect-ps
prefect-ps: ## Show the Prefect server status
	$(COMPOSE) --profile orchestration ps prefect-server


.PHONY: prefect-logs
prefect-logs: ## Follow Prefect server logs
	$(COMPOSE) --profile orchestration logs --follow prefect-server


.PHONY: prefect-down
prefect-down: ## Stop and remove the Prefect stack
	$(COMPOSE) --profile orchestration down

.PHONY: prefect-pool
prefect-pool: ## Create or update the local Prefect process pool
	$(UV) run prefect work-pool create --type process --overwrite local-process-pool


.PHONY: prefect-deploy
prefect-deploy: ## Register all deployments from prefect.yaml
	$(UV) run prefect deploy --all


.PHONY: prefect-worker
prefect-worker: ## Start a local worker for training deployments
	$(UV) run prefect worker start --pool local-process-pool

.PHONY: mlflow-up
mlflow-up: ## Build and start the local MLflow server
	$(COMPOSE) --profile tracking up --detach --build mlflow


.PHONY: mlflow-ps
mlflow-ps: ## Show the MLflow server status
	$(COMPOSE) --profile tracking ps mlflow


.PHONY: mlflow-logs
mlflow-logs: ## Follow MLflow server logs
	$(COMPOSE) --profile tracking logs --follow mlflow


.PHONY: mlflow-down
mlflow-down: ## Stop and remove the MLflow stack
	$(COMPOSE) --profile tracking down

.PHONY: monitoring-config
monitoring-config: ## Validate the monitoring Compose configuration
	$(COMPOSE) --profile monitoring config


.PHONY: monitoring-up
monitoring-up: ## Start the complete local monitoring stack
	$(COMPOSE) --profile monitoring --profile tracking up --detach api mlflow dashboard prometheus alertmanager grafana


.PHONY: monitoring-rebuild
monitoring-rebuild: ## Rebuild and recreate the complete monitoring stack
	$(COMPOSE) --profile monitoring --profile tracking up --detach --build --force-recreate api mlflow dashboard prometheus alertmanager grafana


.PHONY: monitoring-ps
monitoring-ps: ## Show monitoring service status
	$(COMPOSE) --profile monitoring --profile tracking ps api mlflow dashboard prometheus alertmanager grafana


.PHONY: monitoring-logs
monitoring-logs: ## Follow monitoring service logs
	$(COMPOSE) --profile monitoring --profile tracking logs --follow api mlflow dashboard prometheus alertmanager grafana


.PHONY: monitoring-down
monitoring-down: ## Stop and remove monitoring services
	$(COMPOSE) --profile monitoring --profile tracking rm --force --stop mlflow dashboard prometheus alertmanager grafana


.PHONY: dashboard-up
dashboard-up: ## Build and start API, MLflow and operations dashboard
	$(COMPOSE) --profile monitoring --profile tracking up --detach --build api mlflow dashboard


.PHONY: dashboard-ps
dashboard-ps: ## Show API, MLflow and dashboard status
	$(COMPOSE) --profile monitoring --profile tracking ps api mlflow dashboard


.PHONY: dashboard-logs
dashboard-logs: ## Follow operations dashboard logs
	$(COMPOSE) --profile monitoring --profile tracking logs --follow dashboard


.PHONY: dashboard-down
dashboard-down: ## Stop and remove the operations dashboard
	$(COMPOSE) --profile monitoring --profile tracking rm --force --stop dashboard

.PHONY: monitoring-validate
monitoring-validate: ## Validate Prometheus and Alertmanager configuration
	$(COMPOSE) --profile monitoring run --rm --no-deps --entrypoint /bin/promtool prometheus check config /etc/prometheus/prometheus.yml
	$(COMPOSE) --profile monitoring run --rm --no-deps --entrypoint /bin/amtool alertmanager check-config /etc/alertmanager/alertmanager.yml
.PHONY: terraform-fmt
terraform-fmt: ## Check Terraform formatting
	terraform -chdir=infrastructure/terraform-bootstrap fmt -check -recursive
	terraform -chdir=infrastructure/terraform fmt -check -recursive


.PHONY: terraform-init
terraform-init: ## Initialize Terraform without a remote backend
	terraform -chdir=infrastructure/terraform-bootstrap init -backend=false
	terraform -chdir=infrastructure/terraform init -backend=false


.PHONY: terraform-validate
terraform-validate: terraform-init terraform-fmt ## Validate Terraform modules
	terraform -chdir=infrastructure/terraform-bootstrap validate
	terraform -chdir=infrastructure/terraform validate