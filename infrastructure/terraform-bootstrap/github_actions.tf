locals {
  github_repository = join(
    "/",
    [
      var.github_repository_owner,
      var.github_repository_name,
    ],
  )

  github_actions_roles = toset([
    "roles/artifactregistry.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountUser",
    "roles/run.admin",
    "roles/secretmanager.admin",
    "roles/serviceusage.serviceUsageAdmin",
    "roles/storage.admin",
  ])
}

resource "google_project_service" "github_actions" {
  for_each = var.enable_github_actions ? toset([
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ]) : toset([])

  project            = var.gcp_project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_iam_workload_identity_pool" "github_actions" {
  count = var.enable_github_actions ? 1 : 0

  project = var.gcp_project_id

  workload_identity_pool_id = format(
    "gha-%s-pool",
    substr(local.project_slug, 0, 18),
  )

  display_name = "GitHub Actions"

  depends_on = [
    google_project_service.github_actions,
  ]
}

resource "google_iam_workload_identity_pool_provider" "github_actions" {
  count = var.enable_github_actions ? 1 : 0

  project = var.gcp_project_id

  workload_identity_pool_id = (
    google_iam_workload_identity_pool
    .github_actions[0]
    .workload_identity_pool_id
  )

  workload_identity_pool_provider_id = "github"

  display_name = "GitHub repository provider"

  attribute_mapping = {
    "google.subject"             = "assertion.sub"
    "attribute.repository"       = "assertion.repository"
    "attribute.repository_owner" = "assertion.repository_owner"
  }

  attribute_condition = format(
    "assertion.repository == '%s'",
    local.github_repository,
  )

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "github_actions" {
  count = var.enable_github_actions ? 1 : 0

  project = var.gcp_project_id

  account_id = format(
    "gha-%s-dep",
    substr(local.project_slug, 0, 18),
  )

  display_name = "GitHub Actions deployer"
}

resource "google_service_account_iam_member" "github_identity" {
  count = var.enable_github_actions ? 1 : 0

  service_account_id = (
    google_service_account
    .github_actions[0]
    .name
  )

  role = "roles/iam.workloadIdentityUser"

  member = format(
    "principalSet://iam.googleapis.com/%s/attribute.repository/%s",
    google_iam_workload_identity_pool.github_actions[0].name,
    local.github_repository,
  )
}

resource "google_project_iam_member" "github_actions" {
  for_each = (
    var.enable_github_actions
    ? local.github_actions_roles
    : toset([])
  )

  project = var.gcp_project_id
  role    = each.value
  member = format(
    "serviceAccount:%s",
    google_service_account.github_actions[0].email,
  )
}

resource "google_storage_bucket_iam_member" "github_state" {
  count = var.enable_github_actions ? 1 : 0

  bucket = google_storage_bucket.terraform_state.name
  role   = "roles/storage.objectAdmin"

  member = format(
    "serviceAccount:%s",
    google_service_account.github_actions[0].email,
  )
}