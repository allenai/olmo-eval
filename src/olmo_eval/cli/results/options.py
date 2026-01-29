"""Shared CLI options and decorators for results commands."""

from __future__ import annotations

import functools
from typing import Any

import click

from olmo_eval.cli.utils import console


def s3_options(func: Any) -> Any:
    """Decorator that adds common S3 connection options to a command."""

    @click.option(
        "--s3-endpoint-url",
        envvar="S3_ENDPOINT_URL",
        default=None,
        help="S3 endpoint URL (for LocalStack or S3-compatible services).",
    )
    @click.option(
        "--s3-region",
        envvar="AWS_REGION",
        default="us-east-1",
        help="AWS region.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def db_options(func: Any) -> Any:
    """Decorator that adds common database connection options to a command."""

    @click.option(
        "--db-host",
        envvar="OLMO_EVAL_DB_HOST",
        default="localhost",
        help="Database host.",
    )
    @click.option(
        "--db-port",
        envvar="OLMO_EVAL_DB_PORT",
        default=5432,
        type=int,
        help="Database port.",
    )
    @click.option(
        "--db-name",
        envvar="OLMO_EVAL_DB_NAME",
        default="olmo_eval",
        help="Database name.",
    )
    @click.option(
        "--db-user",
        envvar="OLMO_EVAL_DB_USER",
        default="postgres",
        help="Database user.",
    )
    @click.option(
        "--db-password",
        envvar="OLMO_EVAL_DB_PASSWORD",
        default="postgres",
        help="Database password.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def get_database_session(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> Any:
    """Create and initialize a DatabaseSession.

    Returns:
        Initialized DatabaseSession instance.

    Raises:
        SystemExit: If psycopg is not installed.
    """
    try:
        from olmo_eval.storage.backends.postgres.session import (
            get_database_session as _get_database_session,
        )

        return _get_database_session(db_host, db_port, db_name, db_user, db_password)
    except ImportError:
        console.print(
            "[red]Error:[/red] Database support requires psycopg. "
            "Install with: pip install psycopg[binary]"
        )
        raise SystemExit(1) from None
