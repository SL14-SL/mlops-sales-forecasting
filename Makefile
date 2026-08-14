include .env
export

# --- Configuration ---
.DEFAULT_GOAL := help
PYTHON_VERSION := 3.12.9

LOCAL_MLFLOW_TRACKING_URI := http://localhost:5000
LOCAL_PREFECT_API_URL := http://localhost:4221/api
PREFECT_POOL ?= local-pool
PREFECT_PROJECT_DIR ?= $(CURDIR)

.PHONY: help setup dev-up dev-down dev train train-force test lint clean \
        ui-prefect ui-mlflow ui-dashboard prefect-status wait-prefect logs \
        refresh-api prefect-pool prefect-setup prefect-worker auto-retrain \
        snapshot-demo-baseline reset-lifecycle-run \
        demo-promo-without-retraining demo-promo-with-retraining \
        controlled-retraining-experiment train-bootstrap \
		list-serving-releases rollback-serving

# --- Main Entry Point ---

all: setup dev-up wait-prefect prefect-pool prefect-setup train-bootstrap test ## Run the complete pipeline
	@echo "✨ Full build successful! API, MLflow, and Prefect are running."

help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Environment Setup ---

setup: ## Initialize local virtual environment using uv
	@echo "🚀 Initializing project with Python $(PYTHON_VERSION)..."
	uv venv --python $(PYTHON_VERSION) --allow-existing
	uv pip install -e .
	@echo "✅ Setup complete. Use 'source .venv/bin/activate' to start."

# --- Docker & Infrastructure ---

dev-up: ## Spin up the full stack (DB, MLflow, API, Prefect) in detached mode
	@echo "🐳 Starting container stack..."
	mkdir -p mlruns
	mkdir -p models
	mkdir -p results
	mkdir -p prefect_data
	mkdir -p data/raw/new_batches
	mkdir -p data/raw/quarantine
	mkdir -p data/features
	mkdir -p data/splits
	mkdir -p data/validation
	mkdir -p data/predictions/archive
	mkdir -p data/monitoring
	UID=$$(id -u) GID=$$(id -g) docker compose up -d --build
	@echo "✅ Services are live: API (8000), Streamlit (8501), MLflow (5000), Prefect (4221), Grafana (3000), Prometheus (9090)"

dev-down: ## Stop all containers and remove networks
	@echo "🛑 Shutting down services..."
	docker compose down

dev: dev-up wait-prefect prefect-pool prefect-setup ## Start local stack and register Prefect deployment
	@echo "✅ Dev environment ready. Start worker with 'make prefect-worker'."

logs: ## Follow logs from the API service
	docker compose logs -f api

refresh-api: ## Restart or recreate API service using Docker Compose
	@echo "🔄 Refreshing API..."
	docker compose up -d api

# --- Prefect Specifics ---

prefect-status: ## Check local Prefect server and configuration
	@echo "🔍 Checking Prefect server status..."
	@PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		uv run --active prefect config view
	@PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		uv run --active prefect work-pool ls
	@curl -s "$(LOCAL_PREFECT_API_URL)/health" || \
		echo "⚠️ Prefect server is not reachable. Run 'make dev-up'."

wait-prefect: ## Wait until Prefect server is reachable
	@echo "⏳ Waiting for Prefect server ($(LOCAL_PREFECT_API_URL)/health)..."
	@until curl -s "$(LOCAL_PREFECT_API_URL)/health" > /dev/null; do \
		sleep 2; \
		echo "Prefect not ready yet..."; \
	done
	@echo "✅ Prefect is online!"

prefect-pool: wait-prefect ## Create local Prefect work pool if missing
	@echo "🏊 Ensuring Prefect work pool '$(PREFECT_POOL)' exists..."
	@PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		uv run --active prefect work-pool inspect "$(PREFECT_POOL)" \
		> /dev/null 2>&1 || \
		PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		uv run --active prefect work-pool create \
			--type process \
			"$(PREFECT_POOL)"

prefect-setup: wait-prefect prefect-pool ## Register/update local Prefect deployment
	@echo "🧭 Registering Prefect deployment..."
	@APP_ENV=dev \
		PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		PREFECT_API_KEY= \
		PREFECT_PROJECT_DIR="$(PREFECT_PROJECT_DIR)" \
		MLFLOW_TRACKING_URI="$(LOCAL_MLFLOW_TRACKING_URI)" \
		uv run --active prefect deploy \
			flows/auto_retrain_flow.py:auto_retrain_flow \
			--name auto-retrain \
			--pool "$(PREFECT_POOL)"

prefect-worker: wait-prefect prefect-pool ## Start Prefect worker for local pool
	@echo "👷 Starting Prefect worker for pool '$(PREFECT_POOL)'..."
	APP_ENV=dev \
		PREFECT_API_URL="$(LOCAL_PREFECT_API_URL)" \
		PREFECT_API_KEY= \
		PREFECT_PROJECT_DIR="$(PREFECT_PROJECT_DIR)" \
		MLFLOW_TRACKING_URI="$(LOCAL_MLFLOW_TRACKING_URI)" \
		uv run --active prefect worker start \
			--pool "$(PREFECT_POOL)"

# --- UI Quicklinks ---

ui-prefect: ## Open Prefect UI in the browser
	@python3 -m webbrowser http://localhost:4200

ui-mlflow: ## Open MLflow UI in the browser
	@python3 -m webbrowser http://localhost:5000


COMPOSE_RUN_API=docker compose exec -T \
	-e APP_ENV=dev \
	-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
	-e PREFECT_API_URL=http://prefect:4200/api \
	-e PREDICTION_API_URL=http://api:8080/predict \
	api

# --- ML Pipeline Tasks ---

train: wait-prefect ## Execute the training flow inside the API container
	@echo "🧠 Starting training flow inside API container..."
	$(COMPOSE_RUN_API) uv run python flows/training_flow.py

train-force: wait-prefect ## Execute forced training flow inside the API container
	@echo "🧠 Starting forced training flow inside API container..."
	$(COMPOSE_RUN_API) uv run python flows/training_flow.py --force

train-bootstrap: wait-prefect ## Create the initial Champion in an empty registry
	@echo "🌱 Bootstrapping initial Champion..."
	$(COMPOSE_RUN_API) uv run python flows/training_flow.py --force --bootstrap

auto-retrain: wait-prefect ## Run auto retrain flow once manually inside the API container
	@echo "🤖 Running auto retrain flow once inside API container..."
	$(COMPOSE_RUN_API) uv run python flows/auto_retrain_flow.py

predict-test: ## Send a sample prediction request and format output
	@echo "🧪 Sending test prediction request..."
	@curl -fsS -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-H "X-API-KEY: $(API_KEY)" \
		-d '{"inputs":[{"Store":1,"DayOfWeek":5,"Date":"2015-07-31","Open":1,"Customers":500,"Promo":1,"StateHoliday":"0","SchoolHoliday":0}]}' \
		| jq .

demo-forecasting-lifecycle: wait-prefect ## Run forecasting lifecycle demo inside the API container
	@echo "📈 Running forecasting lifecycle demo inside API container..."
	$(COMPOSE_RUN_API) uv run --no-sync python scripts/run_performance_demo.py

# --- Production Helpers ---

GCP_PROJECT_ID ?= $(shell gh variable get GCP_PROJECT_ID 2>/dev/null)
GCP_BUCKET_NAME ?= $(shell gh variable get GCP_BUCKET_NAME 2>/dev/null)
MLFLOW_URL ?= $(shell gh variable get MLFLOW_URL 2>/dev/null)
PREDICTION_API_URL ?= $(shell gh variable get PREDICTION_API_URL 2>/dev/null)

upload-raw-prod: ## Upload raw forecasting data to the production GCS bucket
	@echo "☁️ Uploading raw data to gs://$(GCP_BUCKET_NAME)/data/raw/"
	gcloud storage cp data/raw/train.csv data/raw/store.csv data/raw/test.csv \
		gs://$(GCP_BUCKET_NAME)/data/raw/
	@echo "✅ Raw data uploaded."
	gcloud storage ls gs://$(GCP_BUCKET_NAME)/data/raw/


train-force-prod: ## Execute forced training flow against production cloud services
	@echo "🧠 Starting forced production training flow..."
	PYTHONPATH=. \
	APP_ENV=prod \
	PREFECT_API_URL="$(PREFECT_API_URL)" \
	PREFECT_API_KEY="$(PREFECT_API_KEY)" \
	MLFLOW_TRACKING_URI="$(MLFLOW_URL)" \
	PREDICTION_API_URL="$(PREDICTION_API_URL)" \
	GCP_BUCKET_NAME="$(GCP_BUCKET_NAME)" \
	GCP_PROJECT_ID="$(GCP_PROJECT_ID)" \
	API_KEY="$(API_KEY)" \
	uv run --active python flows/training_flow.py --force


demo-forecasting-lifecycle-prod: ## Run forecasting lifecycle demo against production API and GCS
	@echo "📈 Running forecasting lifecycle demo against production cloud services..."
	PYTHONPATH=. \
	APP_ENV=prod \
	PREFECT_API_URL="$(PREFECT_API_URL)" \
	PREFECT_API_KEY="$(PREFECT_API_KEY)" \
	MLFLOW_TRACKING_URI="$(MLFLOW_URL)" \
	PREDICTION_API_URL="$(PREDICTION_API_URL)" \
	GCP_BUCKET_NAME="$(GCP_BUCKET_NAME)" \
	GCP_PROJECT_ID="$(GCP_PROJECT_ID)" \
	API_KEY="$(API_KEY)" \
	uv run --active python scripts/run_performance_demo.py

debug-prod-env: ## Show production environment values loaded by Make
	@echo "PREFECT_API_URL=$(PREFECT_API_URL)"
	@echo "MLFLOW_URL=$(MLFLOW_URL)"
	@echo "PREDICTION_API_URL=$(PREDICTION_API_URL)"
	@echo "GCP_PROJECT_ID=$(GCP_PROJECT_ID)"
	@echo "GCP_BUCKET_NAME=$(GCP_BUCKET_NAME)"


# --- Quality Assurance ---

test: ## Run unit and integration tests
	@echo "🧪 Running pytest suite..."
	uv run --active pytest tests/

lint: ## Check code style and quality (Ruff)
	@echo "✨ Linting code..."
	uv run --active ruff check .

# --- Cleanup ---

clean: ## Remove temporary files and caches
	@echo "🧹 Cleaning up Python caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✨ Workspace is clean."

clean-venv: ## Remove the virtual environment
	@echo "🗑️ Removing .venv..."
	rm -rf .venv

clean-data: ## Remove local data folders
	@echo "📂 Removing local runtime data folders..."
	rm -rf ./prefect_data ./mlruns ./models 
	@echo "✅ Runtime data folders removed."

clean-all: clean dev-down clean-venv clean-data ## Deep clean everything
	@echo "🐳 Pruning Docker system..."
	docker system prune -f
	docker volume prune -f
	@echo "🧼 Deep clean finished. System is fresh."

	
reset-demo: ## Reset generated artifacts and runtime state, keep only raw input data
	@echo "♻️ Resetting demo state (keeping data/raw)..."
	docker compose down -v
	rm -rf ./mlruns
	rm -rf ./mlruns_artifacts
	rm -rf ./models
	rm -rf ./data/features/*
	rm -rf ./data/splits/*
	rm -rf ./data/validation/*
	rm -rf ./data/predictions/*
	rm -rf ./data/monitoring/*
	rm -rf ./data/versioning/*
	rm -f ./data/raw/simulation_ground_truth.csv || true
	find ./data/raw/new_batches -mindepth 1 -delete
	find ./data/raw/quarantine -mindepth 1 -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f ./mlflow.db
	docker run --rm -v "$$(pwd):/workspace" alpine sh -c "rm -rf /workspace/prefect_data"
	@echo "✅ Demo state reset complete. Raw source data remains in data/raw/."

snapshot-demo-baseline: ## Save model, data and feature state for reproducible comparison runs
	@echo "📸 Saving controlled experiment baseline..."

	@test -f models/latest_state.json || \
		(echo "❌ models/latest_state.json not found. Run 'make train-force' first."; exit 1)

	@test -f data/raw/train.csv || \
		(echo "❌ data/raw/train.csv not found."; exit 1)

	@test -f data/raw/simulation_ground_truth.csv || \
		(echo "❌ Simulation pool not found. Run the ingestion step first."; exit 1)

	@test -f data/features/features.parquet || \
		(echo "❌ Feature table not found. Run 'make train-force' first."; exit 1)

	@test -f data/features/known_calendar.parquet || \
		(echo "❌ Known calendar not found. Run 'make train-force' first."; exit 1)

	mkdir -p data/demo_baseline

	cp models/latest_state.json \
		data/demo_baseline/latest_state.json

	cp data/raw/train.csv \
		data/demo_baseline/train.csv

	cp data/raw/simulation_ground_truth.csv \
		data/demo_baseline/simulation_ground_truth.csv

	cp data/features/features.parquet \
		data/demo_baseline/features.parquet

	cp data/features/known_calendar.parquet \
		data/demo_baseline/known_calendar.parquet

	@curl -sf http://localhost:8000/health \
		| jq -er '.model_version' \
		> data/demo_baseline/champion_version.txt

	@test -s data/demo_baseline/champion_version.txt || \
		(echo "❌ Could not determine the active champion version."; exit 1)

	@echo "✅ Controlled experiment baseline saved."
	@echo "🏆 Champion version: $$(cat data/demo_baseline/champion_version.txt)"


reset-lifecycle-run: ## Restore the controlled experiment baseline
	@echo "♻️ Restoring controlled experiment baseline..."

	@test -f data/demo_baseline/latest_state.json || \
		(echo "❌ Baseline feature state not found. Run 'make snapshot-demo-baseline' first."; exit 1)

	@test -f data/demo_baseline/train.csv || \
		(echo "❌ Baseline training data not found. Run 'make snapshot-demo-baseline' first."; exit 1)

	@test -f data/demo_baseline/simulation_ground_truth.csv || \
		(echo "❌ Baseline simulation pool not found. Run 'make snapshot-demo-baseline' first."; exit 1)

	@test -f data/demo_baseline/features.parquet || \
		(echo "❌ Baseline feature table not found. Run 'make snapshot-demo-baseline' first."; exit 1)

	@test -f data/demo_baseline/known_calendar.parquet || \
		(echo "❌ Baseline calendar not found. Run 'make snapshot-demo-baseline' first."; exit 1)

	@test -s data/demo_baseline/champion_version.txt || \
		(echo "❌ Baseline champion version not found. Run 'make snapshot-demo-baseline' first."; exit 1)

	rm -rf ./data/predictions/*
	rm -rf ./data/monitoring/*
	rm -rf ./data/validation/*
	rm -rf ./data/splits/*

	find ./data/raw/new_batches -mindepth 1 -delete
	find ./data/raw/quarantine -mindepth 1 -delete

	mkdir -p \
		data/predictions \
		data/monitoring \
		data/validation \
		data/splits \
		data/features

	cp data/demo_baseline/latest_state.json \
		models/latest_state.json

	cp data/demo_baseline/train.csv \
		data/raw/train.csv

	cp data/demo_baseline/simulation_ground_truth.csv \
		data/raw/simulation_ground_truth.csv

	cp data/demo_baseline/features.parquet \
		data/features/features.parquet

	cp data/demo_baseline/known_calendar.parquet \
		data/features/known_calendar.parquet

	@echo "🏆 Restoring champion version $$(cat data/demo_baseline/champion_version.txt)..."

	$(COMPOSE_RUN_API) uv run --no-sync python scripts/set_model_alias.py \
		--alias champion \
		--version "$$(cat data/demo_baseline/champion_version.txt)" \
		--reload-api

	@echo "✅ Controlled experiment baseline restored."

demo-promo-without-retraining: wait-prefect ## Run the controlled static baseline
	@$(MAKE) reset-lifecycle-run

	@echo "📉 Running promo-effect decay without retraining..."

	$(COMPOSE_RUN_API) uv run --no-sync python scripts/run_performance_demo.py \
		--scenario gradual_promo_shift \
		--retraining disabled \
		--output-file results/promo_weighted_without_retraining.csv \
		--drift-start-day 20 \
		--drift-duration-days 14 \
		--maximum-base-uplift 0.0 \
		--maximum-promo-uplift -0.25

	cp data/predictions/inference_log.parquet \
		results/promo_weighted_without_predictions.parquet

	cp data/monitoring/cumulative_ground_truth.csv \
		results/promo_weighted_without_ground_truth.csv

	@echo "✅ Static baseline and row-level artifacts saved."

demo-promo-with-retraining: wait-prefect ## Run the controlled adaptive lifecycle
	@$(MAKE) reset-lifecycle-run

	@echo "📈 Running promo-effect decay with retraining..."

	$(COMPOSE_RUN_API) uv run --no-sync python scripts/run_performance_demo.py \
		--scenario gradual_promo_shift \
		--retraining enabled \
		--output-file results/promo_mild_weights_with_retraining.csv \
		--drift-start-day 20 \
		--drift-duration-days 14 \
		--maximum-base-uplift 0.0 \
		--maximum-promo-uplift -0.25

	cp data/predictions/inference_log.parquet \
		results/promo_mild_weights_with_predictions.parquet

	cp data/monitoring/cumulative_ground_truth.csv \
		results/promo_mild_weights_with_ground_truth.csv

	@echo "✅ Adaptive lifecycle and row-level artifacts saved."

controlled-retraining-experiment: wait-prefect ## Run both controlled variants and generate the final comparison
	@echo "============================================================"
	@echo "🧪 Starting controlled retraining experiment"
	@echo "============================================================"

	@$(MAKE) demo-promo-without-retraining
	@$(MAKE) demo-promo-with-retraining

	@echo "📊 Generating final comparison..."

	uv run --active python scripts/plot_retraining_comparison.py

	@echo "============================================================"
	@echo "✅ Controlled retraining experiment completed"
	@echo "📄 Results are available under results/"
	@echo "============================================================"



list-serving-releases: ## List published serving releases
	@curl -fsS \
		http://localhost:8000/admin/serving-releases \
		-H "X-API-KEY: $(API_KEY)" \
		| jq .

rollback-serving: ## Roll back to RELEASE_ID
	@test -n "$(RELEASE_ID)" || \
		(echo "❌ RELEASE_ID is required."; exit 1)
	@echo "↩️ Rolling back serving release to $(RELEASE_ID)..."
	@curl -fsS -X POST \
		http://localhost:8000/admin/rollback-serving-release \
		-H "Content-Type: application/json" \
		-H "X-API-KEY: $(API_KEY)" \
		-d '{"release_id":"$(RELEASE_ID)"}' \
		| jq .