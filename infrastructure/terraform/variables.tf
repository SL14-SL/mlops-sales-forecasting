variable "gcp_project_id" {
  description = "Google Cloud project used for the deployment."
  type        = string

  validation {
    condition     = length(trimspace(var.gcp_project_id)) > 0
    error_message = "gcp_project_id must not be empty."
  }
}

variable "region" {
  description = "Google Cloud region for regional resources."
  type        = string
  default     = "europe-west1"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"

  validation {
    condition = contains(
      [
        "dev",
        "staging",
        "prod",
      ],
      var.environment,
    )

    error_message = "environment must be dev, staging or prod."
  }
}

variable "artifact_registry_repository" {
  description = "Artifact Registry repository name."
  type        = string
  default     = "mlops-images"
}

variable "storage_location" {
  description = "Location used for the MLOps artifact bucket."
  type        = string
  default     = "EU"
}

variable "force_destroy_artifact_bucket" {
  description = "Allow Terraform to remove a non-empty artifact bucket."
  type        = bool
  default     = false
}

variable "container_image" {
  description = "Complete container image URI deployed to Cloud Run."
  type        = string

  validation {
    condition     = length(trimspace(var.container_image)) > 0
    error_message = "container_image must not be empty."
  }
}

variable "allow_unauthenticated" {
  description = "Allow unauthenticated network access to the API."
  type        = bool
  default     = false
}

variable "minimum_instances" {
  description = "Minimum number of Cloud Run instances."
  type        = number
  default     = 0

  validation {
    condition     = var.minimum_instances >= 0
    error_message = "minimum_instances must not be negative."
  }
}

variable "maximum_instances" {
  description = "Maximum number of Cloud Run instances."
  type        = number
  default     = 3

  validation {
    condition     = var.maximum_instances >= 1
    error_message = "maximum_instances must be at least one."
  }
}

variable "container_cpu" {
  description = "CPU limit assigned to one Cloud Run instance."
  type        = string
  default     = "1"
}

variable "container_memory" {
  description = "Memory limit assigned to one Cloud Run instance."
  type        = string
  default     = "512Mi"
}

variable "mlflow_tracking_uri" {
  description = "Optional external MLflow tracking URI."
  type        = string
  default     = ""
}

variable "prefect_api_url" {
  description = "Optional external Prefect API URL."
  type        = string
  default     = ""
}

variable "deploy_cloud_run" {
  description = "Deploy the Cloud Run API after its image is available."
  type        = bool
  default     = false
}