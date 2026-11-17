variable "aws_region" {
  description = "AWS region containing the project resources."
  type        = string
  default     = "eu-north-1"
}

variable "aws_account_id" {
  description = "Expected AWS account ID."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "aws_account_id must contain exactly 12 digits."
  }
}

variable "project_name" {
  description = "Short resource name prefix."
  type        = string
  default     = "mlops-sales-forecasting"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,29}$", var.project_name))
    error_message = "Use 3 to 30 lowercase letters, digits or hyphens, starting with a letter."
  }
}

variable "fargate_training_image_digest" {
  description = "Digest of the training image containing the Fargate adapter."
  type        = string

  validation {
    condition = can(regex(
      "^sha256:[a-f0-9]{64}$",
      var.fargate_training_image_digest
    ))
    error_message = "Provide a SHA-256 image digest."
  }
}

variable "api_image_digest" {
  description = "Digest of the tested API image with AWS dependencies."
  type        = string

  validation {
    condition     = can(regex("^sha256:[a-f0-9]{64}$", var.api_image_digest))
    error_message = "Provide a SHA-256 image digest."
  }
}

variable "api_desired_count" {
  description = "Number of running API tasks; zero stops the API containers."
  type        = number
  default     = 0

  validation {
    condition     = contains([0, 1], var.api_desired_count)
    error_message = "Use 0 to stop the demo API or 1 to run it."
  }
}

variable "api_serving_enabled" {
  description = "Whether the API service, load balancer and gateway are deployed."
  type        = bool
  default     = false
}

variable "environment" {
  type    = string
  default = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Use dev, staging or prod."
  }
}

variable "mlflow_tracking_uri" {
  description = "Reachable shared MLflow tracking/registry HTTPS endpoint with a durable DB backend."
  type        = string
  default     = ""
  validation {
    condition     = var.mlflow_managed_enabled || can(regex("^https://[^/]+", var.mlflow_tracking_uri))
    error_message = "Set the shared HTTPS MLflow endpoint; job-local SQLite is not a registry."
  }
}

variable "prefect_api_url" {
  description = "Optional reachable Prefect API. Empty uses an ephemeral local Prefect server."
  type        = string
  default     = ""
}

variable "mlflow_token_secret_arn" {
  description = "Optional Secrets Manager ARN containing MLFLOW_TRACKING_TOKEN."
  type        = string
  default     = ""
}

variable "prefect_key_secret_arn" {
  description = "Optional Secrets Manager ARN containing PREFECT_API_KEY."
  type        = string
  default     = ""
}


variable "mlflow_managed_enabled" {
  description = "Provision MLflow, RDS PostgreSQL and an authenticated HTTPS gateway."
  type        = bool
  default     = false
}

variable "mlflow_image_digest" {
  description = "Digest of Dockerfile.mlflow image; required for managed MLflow."
  type        = string
  default     = ""
}

variable "mlflow_desired_count" {
  type    = number
  default = 1
  validation {
    condition     = contains([0, 1], var.mlflow_desired_count)
    error_message = "Use zero or one MLflow task; this deployment is single-instance."
  }
}

variable "mlflow_database_deletion_protection" {
  type    = bool
  default = true
}

variable "mlflow_final_snapshot_identifier" {
  description = "Unique final snapshot name when deleting the database."
  type        = string
  default     = ""
}
