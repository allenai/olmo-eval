# olmo-eval dashboard infrastructure

Terraform for the GCP resources the dashboard needs in the Skiff2 project `ai2-skiff2-olmo-eval`
(region `us-west1`). Skiff2 manages the load balancer, Cloud Run services, IAP, certificates,
Cloud Armor, the `github-actions` service account and Workload Identity Federation. This
configuration does not touch any of those.

State lives in the Skiff2 state bucket `ai2-skiff2-olmo-eval-tf-state` under the prefix
`terraform/olmo-eval`. Skiff2 uses `terraform/infra` and `terraform/services`.

## Resources

| Resource | Name |
|---|---|
| Cloud SQL Postgres 17 instance | `olmo-eval-db` (connection name `ai2-skiff2-olmo-eval:us-west1:olmo-eval-db`) |
| Databases | `olmo_eval` (prod Skiff environment), `olmo_eval_dev` (all other environments) |
| API runtime service account | `olmo-eval-api@ai2-skiff2-olmo-eval.iam.gserviceaccount.com` |
| API IAM database user | `olmo-eval-api@ai2-skiff2-olmo-eval.iam` |
| Results bucket | `gs://ai2-skiff2-olmo-eval-results` (`runs/` for prod, `dev/runs/` for other environments, deleted after 30 days) |
| Build assets bucket | `gs://ai2-skiff2-olmo-eval-build-assets` (Telegraf fonts under `fonts/`) |

The instance has a public IP with no authorized networks and requires the Cloud SQL connectors
(`connector_enforcement = "REQUIRED"`). Clients connect with the Cloud SQL Python Connector and
IAM database authentication. There are no database passwords. The built-in `postgres` user
that Cloud SQL creates has no password set and cannot log in. Deletion protection, automated
backups and point-in-time recovery are on.

## Database privileges

Both IAM database users (the API service account and each entry in `debug_iam_users`) are
created with `database_roles = ["cloudsqlsuperuser"]`. Cloud SQL makes `cloudsqlsuperuser` the
owner of both databases, so its members can create tables in `public`. No manual grants are
needed. The API's first migration sets default privileges so members of `cloudsqlsuperuser` can
read tables the API creates.

Check the privileges after an apply. By default the script impersonates the API service account:

```bash
uv run infra/terraform/scripts/check_db.py
uv run infra/terraform/scripts/check_db.py --as-self --user you@allenai.org
```

## Applying changes

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

You need owner (or equivalent) on `ai2-skiff2-olmo-eval`. CI runs `terraform fmt -check` and
`terraform validate` only. Commit `.terraform.lock.hcl` when the provider version changes, and
lock for every platform CI and developers use:

```bash
terraform providers lock -platform=linux_amd64 -platform=linux_arm64 \
  -platform=darwin_amd64 -platform=darwin_arm64
```

## Fonts

The UI uses the commercial Telegraf font, which must never be committed to this public repo.
Upload new font files to the build assets bucket:

```bash
gcloud storage cp Telegraf-*.otf gs://ai2-skiff2-olmo-eval-build-assets/fonts/
```

The deploy workflow downloads them into `dashboard/ui/public/fonts/` before building the UI image.
