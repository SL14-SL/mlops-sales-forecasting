variable "gcp_project_id" {
  description = "Google Cloud project containing the state bucket."
  type        = string

  validation {
    condition     = length(trimspace(var.gcp_project_id)) > 0
    error_message = "gcp_project_id must not be empty."
  }
}

variable "storage_location" {
  description = "Location used for the Terraform state bucket."
  type        = string
  default     = "EU"
}

variable "enable_github_actions" {
  description = "Create GitHub Actions Workload Identity resources."
  type        = bool
  default     = false
}

variable "github_repository_owner" {
  description = "Owner of the GitHub repository."
  type        = string
  default     = "replace-me"
}

variable "github_repository_name" {
  description = "Name of the GitHub repository."
  type        = string
  default     = "replace-me"
}