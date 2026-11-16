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
  default     = "sha256:8d851e70ca5dc7774dc176f0f829f4fb8094a491d61c4c7f0b26314156553c37"

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
  default     = "sha256:0c72b2d14caef22f9e1230871dcde1bf89c3fe0a84dcd605ed1f6f711755747f"

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