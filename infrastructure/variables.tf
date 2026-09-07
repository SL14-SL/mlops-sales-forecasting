# infrastructure/variables.tf

variable "project_id" {
  description = "The ID of the GCP project where resources are managed"
  type        = string
}

variable "region" {
  description = "The GCP region for all resources"
  type        = string
  default     = "europe-west1"
}

# environment variable to distinguish between dev, staging, and prod later
variable "environment" {
  description = "The environment name (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "github_repo" {
  description = "GitHub repository in format owner/repo"
  type        = string
}

variable "mlflow_database_version" {
  description = "PostgreSQL version used by the MLflow Cloud SQL backend"
  type        = string
  default     = "POSTGRES_15"
}

variable "mlflow_database_tier" {
  description = "Machine tier used by the cost-conscious MLflow database"
  type        = string
  default     = "db-f1-micro"
}

variable "mlflow_database_name" {
  description = "Cloud SQL database used by MLflow"
  type        = string
  default     = "mlflow"
}

variable "mlflow_database_user" {
  description = "PostgreSQL user used by MLflow"
  type        = string
  default     = "mlflow"
}

variable "mlflow_database_disk_size_gb" {
  description = "Initial Cloud SQL disk size in GiB"
  type        = number
  default     = 10

  validation {
    condition     = var.mlflow_database_disk_size_gb >= 10
    error_message = "Cloud SQL disk size must be at least 10 GiB."
  }
}