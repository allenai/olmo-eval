# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[asyncpg]>=1.18",
#     "google-auth>=2.40",
# ]
# ///
"""Check that an IAM database user can connect to Cloud SQL and create objects.

Run after `terraform apply`:

    uv run infra/terraform/scripts/check_db.py                 # as the API service account
    uv run infra/terraform/scripts/check_db.py --as-self --user you@allenai.org  # as your ADC user

The default impersonates the API runtime service account, which needs
roles/iam.serviceAccountTokenCreator on it (granted to var.debug_iam_users). For each database
the script prints the schema and database CREATE privileges, creates and drops a temp table and a
table in public, and runs CREATE EXTENSION IF NOT EXISTS pg_trgm. It exits non-zero on any failure.
"""

import argparse
import asyncio
import sys

import google.auth
from google.auth import impersonated_credentials
from google.cloud.sql.connector import Connector, IPTypes

INSTANCE = "ai2-skiff2-olmo-eval:us-west1:olmo-eval-db"
API_SA = "olmo-eval-api@ai2-skiff2-olmo-eval.iam.gserviceaccount.com"
DATABASES = ["olmo_eval", "olmo_eval_dev"]
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/sqlservice.login",
]


def _credentials(as_self: bool):
    source, _ = google.auth.default(scopes=SCOPES)
    if as_self:
        return source, None
    creds = impersonated_credentials.Credentials(
        source_credentials=source, target_principal=API_SA, target_scopes=SCOPES
    )
    return creds, API_SA.removesuffix(".gserviceaccount.com")


async def _check(connector: Connector, user: str, database: str) -> bool:
    conn = await connector.connect_async(
        INSTANCE, "asyncpg", user=user, db=database, enable_iam_auth=True, ip_type=IPTypes.PUBLIC
    )
    try:
        row = await conn.fetchrow(
            "SELECT current_user, "
            "has_schema_privilege('public', 'CREATE') AS schema_create, "
            "has_database_privilege(current_database(), 'CREATE') AS db_create"
        )
        print(
            f"{database}: user={row['current_user']} schema_create={row['schema_create']} "
            f"db_create={row['db_create']}"
        )
        await conn.execute("CREATE TEMP TABLE check_db_probe (id int)")
        await conn.execute("DROP TABLE check_db_probe")
        await conn.execute("CREATE TABLE public.check_db_probe (id int)")
        await conn.execute("DROP TABLE public.check_db_probe")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        print(f"{database}: temp table, public table and pg_trgm OK")
        return bool(row["schema_create"] and row["db_create"])
    finally:
        await conn.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-self", action="store_true", help="connect as your own ADC user")
    parser.add_argument("--user", help="IAM database user (default: derived from credentials)")
    args = parser.parse_args()

    creds, user = _credentials(args.as_self)
    user = args.user or user
    if user is None:
        parser.error("--as-self needs --user <your email>")

    ok = True
    async with Connector(
        credentials=creds, refresh_strategy="lazy", loop=asyncio.get_running_loop()
    ) as connector:
        for database in DATABASES:
            ok = await _check(connector, user, database) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
