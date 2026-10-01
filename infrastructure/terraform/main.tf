provider "google" {
  project = var.gcp_project_id
  region  = var.region
}

locals {
  project_slug = "mlops-sales-forecasting"
  name_prefix  = "${local.project_slug}-${var.environment}"

  required_services = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])
}

resource "google_project_service" "required" {
  for_each = local.required_services

  project            = var.gcp_project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "containers" {
  project       = var.gcp_project_id
  location      = var.region
  repository_id = var.artifact_registry_repository
  description   = "Container images for ${local.project_slug}."
  format        = "DOCKER"

  depends_on = [
    google_project_service.required[
      "artifactregistry.googleapis.com"
    ],
  ]
}

resource "google_storage_bucket" "artifacts" {
  project                     = var.gcp_project_id
  name                        = "${var.gcp_project_id}-${var.environment}-artifacts"
  location                    = var.storage_location
  uniform_bucket_level_access = true
  force_destroy               = var.force_destroy_artifact_bucket

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
      with_state         = "ARCHIVED"
    }

    action {
      type = "Delete"
    }
  }

  depends_on = [
    google_project_service.required[
      "storage.googleapis.com"
    ],
  ]
}

resource "google_service_account" "api" {
  project = var.gcp_project_id

  account_id = substr(
    "${local.project_slug}-${var.environment}-api",
    0,
    30,
  )

  display_name = "${local.name_prefix} API"

  depends_on = [
    google_project_service.required[
      "iam.googleapis.com"
    ],
  ]
}

resource "google_storage_bucket_iam_member" "api_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.api.email}"
}