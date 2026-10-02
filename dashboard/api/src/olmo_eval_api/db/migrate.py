"""Run alembic migrations at startup under a Postgres advisory lock.

Both services migrate at startup. The lock serializes concurrent instances; the startup
probe keeps traffic away until ``/health`` sees a migrated database.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, text
from sqlalchemy.ext.asyncio import AsyncEngine

ADVISORY_LOCK_ID = 727274001
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def alembic_config(connection: Connection | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def _upgrade_head(connection: Connection) -> None:
    command.upgrade(alembic_config(connection), "head")


async def run_migrations(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(text(f"SELECT pg_advisory_lock({ADVISORY_LOCK_ID})"))
        try:
            await conn.run_sync(_upgrade_head)
            await conn.commit()
        finally:
            await conn.execute(text(f"SELECT pg_advisory_unlock({ADVISORY_LOCK_ID})"))
            await conn.commit()


async def prune_stats_cache(engine: AsyncEngine, days: int = 30) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM stats_cache WHERE created_at < now() - make_interval(days => :d)"),
            {"d": days},
        )
