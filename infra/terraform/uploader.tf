# Identity for Beaker jobs that upload results. It has no Google Cloud roles: the ingest
# service allowlists it (UPLOADER_SERVICE_ACCOUNT in dashboard/api settings), and
# artifacts reach GCS through URLs the API signs. A leaked key can only upload results,
# which any @allenai.org account can already do with its own credentials.
resource "google_service_account" "uploader" {
  account_id   = "olmo-eval-uploader"
  display_name = "olmo-eval results uploader (Beaker jobs)"
}

# A new key is created on the first apply after each rotation period. Jobs launched with
# the old key fail to upload after rotation; their results stay on disk and can be
# re-uploaded with `olmo-eval results upload`.
resource "time_rotating" "uploader_key" {
  rotation_days = 90
}

# The private key is also stored in this configuration's state, which only project
# owners can read. They can create keys for this account anyway.
resource "google_service_account_key" "uploader" {
  service_account_id = google_service_account.uploader.name

  keepers = {
    rotation = time_rotating.uploader_key.id
  }

  lifecycle {
    create_before_destroy = true
  }
}

# olmo-eval beaker launch reads this with the launching user's local credentials and
# copies it into the job's workspace as the olmo_eval_uploader_key Beaker secret.
resource "google_secret_manager_secret" "uploader_key" {
  secret_id = "olmo-eval-uploader-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "uploader_key" {
  secret      = google_secret_manager_secret.uploader_key.id
  secret_data = base64decode(google_service_account_key.uploader.private_key)
}

resource "google_secret_manager_secret_iam_member" "uploader_key_readers" {
  for_each  = toset(var.uploader_key_readers)
  secret_id = google_secret_manager_secret.uploader_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
}

# The launch bills its Secret Manager call to this project (quota project), which needs
# serviceusage.services.use here regardless of the user's own quota project.
resource "google_project_iam_member" "uploader_key_readers_quota" {
  for_each = toset(var.uploader_key_readers)
  project  = var.project_id
  role     = "roles/serviceusage.serviceUsageConsumer"
  member   = each.value
}
