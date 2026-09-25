provider "google" {
  project = var.gcp_project_id
}

locals {
  project_slug = "mlops-sales-forecasting"
}

resource "google_project_service" "storage" {
  project            = var.gcp_project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_storage_bucket" "terraform_state" {
  project  = var.gcp_project_id
  name     = "${var.gcp_project_id}-${local.project_slug}-tfstate"
  location = var.storage_location

  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [
    google_project_service.storage,
  ]
}