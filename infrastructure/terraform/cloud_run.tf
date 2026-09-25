locals {
  optional_runtime_environment = merge(
    var.mlflow_tracking_uri != "" ? {
      MLFLOW_TRACKING_URI = var.mlflow_tracking_uri
    } : {},
    var.prefect_api_url != "" ? {
      PREFECT_API_URL = var.prefect_api_url
    } : {},
  )

  runtime_environment = merge(
    {
      APP_ENV         = var.environment
      GCS_BUCKET_NAME = google_storage_bucket.artifacts.name
      LOG_LEVEL       = "INFO"
    },
    local.optional_runtime_environment,
  )
}

resource "google_secret_manager_secret" "api_key" {
  project   = var.gcp_project_id
  secret_id = "${local.name_prefix}-api-key"

  replication {
    auto {}
  }

  depends_on = [
    google_project_service.required[
      "secretmanager.googleapis.com"
    ],
  ]
}

resource "google_secret_manager_secret_iam_member" "api_access" {
  project   = var.gcp_project_id
  secret_id = google_secret_manager_secret.api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service" "api" {
  count = var.deploy_cloud_run ? 1 : 0

  project  = var.gcp_project_id
  name     = "${local.name_prefix}-api"
  location = var.region

  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.api.email
    timeout         = "300s"

    scaling {
      min_instance_count = var.minimum_instances
      max_instance_count = var.maximum_instances
    }

    containers {
      image = var.container_image

      ports {
        name           = "http1"
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = var.container_cpu
          memory = var.container_memory
        }

        cpu_idle = true
      }

      dynamic "env" {
        for_each = local.runtime_environment

        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name = "API_KEY"

        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        initial_delay_seconds = 0
        timeout_seconds       = 3
        period_seconds        = 5
        failure_threshold     = 12

        http_get {
          path = "/livez"
          port = 8000
        }
      }

      liveness_probe {
        initial_delay_seconds = 10
        timeout_seconds       = 3
        period_seconds        = 30
        failure_threshold     = 3

        http_get {
          path = "/livez"
          port = 8000
        }
      }
    }
  }

  depends_on = [
    google_project_service.required[
      "run.googleapis.com"
    ],
    google_secret_manager_secret_iam_member.api_access,
    google_storage_bucket_iam_member.api_artifacts,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count = (
    var.deploy_cloud_run
    && var.allow_unauthenticated
  ) ? 1 : 0

  project = var.gcp_project_id
  location = (
    google_cloud_run_v2_service.api[0].location
  )
  name = (
    google_cloud_run_v2_service.api[0].name
  )
  role   = "roles/run.invoker"
  member = "allUsers"
}