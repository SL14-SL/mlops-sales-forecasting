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