output "terraform_state_bucket" {
  description = "GCS bucket used for the primary Terraform state."
  value       = google_storage_bucket.terraform_state.name
}

output "github_workload_identity_provider" {
  description = "Provider identifier used by GitHub Actions."
  value = try(
    google_iam_workload_identity_pool_provider
    .github_actions[0]
    .name,
    null,
  )
}

output "github_deployer_service_account" {
  description = "Service account impersonated by GitHub Actions."
  value = try(
    google_service_account.github_actions[0].email,
    null,
  )
}