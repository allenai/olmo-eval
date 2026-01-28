"""Alembic environment configuration for olmo-eval-internal."""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import the Base from your models to get metadata
from olmo_eval.storage.db.models import Base

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata


def get_url_from_env():
    """Get database URL from environment variable if available.

    Priority:
    1. OLMO_EVAL_DB_URL environment variable
    2. Standard PostgreSQL env vars (PGHOST, PGUSER, etc.)
    3. alembic.ini sqlalchemy.url

    Returns:
        Tuple of (database URL string or None, source description).
    """
    import logging
    import os

    log = logging.getLogger("alembic.env")

    # Check for full URL in environment
    db_url = os.environ.get("OLMO_EVAL_DB_URL")
    if db_url:
        log.info("Database URL source: OLMO_EVAL_DB_URL environment variable")
        return db_url

    # Check if any PG* vars are set
    host = os.environ.get("PGHOST")
    if not host:
        log.info("Database URL source: alembic.ini sqlalchemy.url")
        return None  # Use alembic.ini default

    # Build URL from standard PostgreSQL env vars
    port = os.environ.get("PGPORT", "5432")
    database = os.environ.get("PGDATABASE", "olmo_eval")
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "postgres")

    log.info("Database URL source: PG* environment variables (PGHOST, PGPORT, etc.)")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url_from_env() or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect server default changes
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Get database URL (from environment or config)
    url = get_url_from_env() or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("No database URL configured. Set OLMO_EVAL_DB_URL or sqlalchemy.url")

    # Update config with URL
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Don't pool connections for migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Detect column type changes
            compare_server_default=True,  # Detect server default changes
            # Optionally render item names with quotes
            # render_as_batch=True for SQLite, not needed for PostgreSQL
        )

        with context.begin_transaction():
            context.run_migrations()


# Determine if we're in offline or online mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
