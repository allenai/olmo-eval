"""Alembic environment configuration for results database."""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from olmo_eval.storage.backends.postgres.db_url import build_results_db_url
from olmo_eval.storage.backends.postgres.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str | None:
    """Get database URL for results database."""
    return build_results_db_url()


def run_migrations_offline() -> None:
    url = get_url() or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = get_url() or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("No database URL configured")

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
