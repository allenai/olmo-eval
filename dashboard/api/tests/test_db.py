"""Migrations, model/migration drift, and GCS signing with a mocked client."""

from __future__ import annotations

from typing import Any

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text

from olmo_eval_api.db.migrate import run_migrations
from olmo_eval_api.db.models import metadata
from olmo_eval_api.storage.gcs import GcsStorage


async def test_migrations_are_idempotent(engine: Any) -> None:
    await run_migrations(engine)
    async with engine.connect() as conn:
        version = (await conn.execute(text("SELECT version_num FROM alembic_version"))).scalar()
        assert version == "0001"
        ext = (
            await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'"))
        ).scalar()
        assert ext == 1


async def test_models_match_migrations(engine: Any) -> None:
    def diff(sync_conn: Any) -> list:
        ctx = MigrationContext.configure(sync_conn, opts={"compare_type": True})
        return compare_metadata(ctx, metadata)

    async with engine.connect() as conn:
        changes = await conn.run_sync(diff)
    assert changes == []


class FakeBlob:
    def __init__(self, name: str, calls: list) -> None:
        self.name = name
        self.calls = calls

    def generate_signed_url(self, **kwargs: Any) -> str:
        self.calls.append((self.name, kwargs))
        return f"https://storage.googleapis.com/bucket/{self.name}?X-Goog-Signature=x"


class FakeBucket:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self.calls)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list = []

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self.calls)


class FakeCreds:
    token = "runtime-token"
    valid = True
    expiry = None


async def test_gcs_signing_uses_signblob() -> None:
    client = FakeClient()
    storage = GcsStorage("bucket", "olmo-eval-api@proj.iam.gserviceaccount.com", client=client)
    storage._credentials = FakeCreds()
    signed = await storage.sign_upload(
        "dev/runs/abc/metrics.json", "application/json", "x" * 22 + "=="
    )
    assert signed.headers == {"Content-Type": "application/json", "Content-MD5": "x" * 22 + "=="}
    name, kwargs = client.calls[0]
    assert name == "dev/runs/abc/metrics.json"
    assert kwargs["version"] == "v4" and kwargs["method"] == "PUT"
    assert kwargs["service_account_email"] == "olmo-eval-api@proj.iam.gserviceaccount.com"
    assert kwargs["access_token"] == "runtime-token"
    assert kwargs["content_md5"] == "x" * 22 + "=="
    assert kwargs["expiration"].total_seconds() == 3600

    download = await storage.sign_download("dev/runs/abc/metrics.json")
    _, kwargs = client.calls[1]
    assert kwargs["method"] == "GET"
    assert kwargs["expiration"].total_seconds() == 900
    assert kwargs["response_disposition"] == 'attachment; filename="metrics.json"'
    assert download.url.startswith("https://storage.googleapis.com/")
