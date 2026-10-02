# Runtime identity for both Cloud Run services (ui with its api sidecar, and ingest).
resource "google_service_account" "api" {
  account_id   = "olmo-eval-api"
  display_name = "olmo-eval dashboard API"
}

locals {
  api_member = "serviceAccount:${google_service_account.api.email}"

  api_project_roles = [
    "roles/cloudsql.client",
    "roles/cloudsql.instanceUser",
    # Skiff reads global-<container>-* secrets at deploy time. Harmless when none exist.
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
  ]

  debug_project_roles = [
    "roles/cloudsql.client",
    "roles/cloudsql.instanceUser",
  ]

  debug_project_bindings = {
    for pair in setproduct(var.debug_iam_users, local.debug_project_roles) :
    "${pair[0]} ${pair[1]}" => { user = pair[0], role = pair[1] }
  }
}

resource "google_project_iam_member" "api" {
  for_each = toset(local.api_project_roles)
  project  = var.project_id
  role     = each.value
  member   = local.api_member
}

# Lets the API sign V4 upload/download URLs through IAM signBlob as itself.
resource "google_service_account_iam_member" "api_self_token_creator" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = local.api_member
}

resource "google_storage_bucket_iam_member" "api_results_object_admin" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = local.api_member
}

resource "google_storage_bucket_iam_member" "api_results_bucket_reader" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.legacyBucketReader"
  member = local.api_member
}

# Impersonation of the API service account, for debugging and verification.
resource "google_service_account_iam_member" "debug_token_creator" {
  for_each           = toset(var.debug_iam_users)
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "user:${each.value}"
}

resource "google_project_iam_member" "debug" {
  for_each = local.debug_project_bindings
  project  = var.project_id
  role     = each.value.role
  member   = "user:${each.value.user}"
}

# CI downloads the Telegraf fonts from the build-assets bucket before building the UI image.
resource "google_storage_bucket_iam_member" "github_actions_build_assets_viewer" {
  bucket = google_storage_bucket.build_assets.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.github_actions_service_account}"
}
