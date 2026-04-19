output "project_id" {
  value = var.project_id
}

output "artifacts_bucket_name" {
  value = google_storage_bucket.artifacts_bucket.name
}

# Full path to the repository for Docker login/push
output "container_registry_uri" {
  description = "Full URI for Docker push"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mlops_repo.name}"
}

output "mlflow_url" {
  value = google_cloud_run_v2_service.mlflow_server.uri
}

output "service_account_email" {
  value = google_service_account.mlops_sa.email
}

output "prediction_api_url" {
  value = google_cloud_run_v2_service.prediction_api.uri
}

# Output the provider name for GitHub Secrets
output "workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github_provider.name
}
