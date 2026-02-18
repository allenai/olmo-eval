"""Shared CLI options and decorators for results commands."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import click

from olmo_eval.cli.utils import console

logger = logging.getLogger(__name__)

# Database connection defaults
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = "olmo_eval"
DEFAULT_DB_USER = "postgres"


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
    """Decorator that adds common database connection options to a command.

    Uses OLMO_EVAL_DB_* environment variables.
    On authentication failure, automatically retries using OLMO_EVAL_DB_SECRET_ARN
    to fetch the password from AWS Secrets Manager.
    """

    @click.option(
        "--db-host",
        envvar="OLMO_EVAL_DB_HOST",
        default=None,
        help=f"Database host (default: {DEFAULT_DB_HOST}).",
    )
    @click.option(
        "--db-port",
        envvar="OLMO_EVAL_DB_PORT",
        default=None,
        type=int,
        help=f"Database port (default: {DEFAULT_DB_PORT}).",
    )
    @click.option(
        "--db-name",
        envvar="OLMO_EVAL_DB_NAME",
        default=None,
        help=f"Database name (default: {DEFAULT_DB_NAME}).",
    )
    @click.option(
        "--db-user",
        envvar="OLMO_EVAL_DB_USER",
        default=None,
        help=f"Database user (default: {DEFAULT_DB_USER}).",
    )
    @click.option(
        "--db-password",
        envvar="OLMO_EVAL_DB_PASSWORD",
        default=None,
        help="Database password. If not set and auth fails, OLMO_EVAL_DB_SECRET_ARN is used.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def _is_auth_failure(error: Exception) -> bool:
    """Check if error is a database authentication failure."""
    msg = str(error).lower()
    return "password authentication failed" in msg or "authentication failed" in msg


def _fetch_password_from_aws(arn: str) -> str:
    """Fetch database password from AWS Secrets Manager.

    Args:
        arn: The ARN of the secret containing the password.

    Returns:
        The password string.

    Raises:
        click.ClickException: If fetching fails.
    """
    try:
        from olmo_eval.launch.beaker.secrets import get_aws_secret_value

        logger.info("Fetching database password from AWS Secrets Manager")
        return get_aws_secret_value(arn)
    except Exception as e:
        raise click.ClickException(f"Failed to fetch password from AWS Secrets Manager: {e}") from e


def get_database_session(
    db_host: str | None,
    db_port: int | None,
    db_name: str | None,
    db_user: str | None,
    db_password: str | None,
) -> Any:
    """Create and initialize a DatabaseSession with auth retry.

    On authentication failure, automatically retries using OLMO_EVAL_DB_SECRET_ARN
    environment variable to fetch the password from AWS Secrets Manager.

    Args:
        db_host: Database host (default: localhost).
        db_port: Database port (default: 5432).
        db_name: Database name (default: olmo_eval).
        db_user: Database user (default: postgres).
        db_password: Database password.

    Returns:
        Initialized DatabaseSession instance.

    Raises:
        SystemExit: If psycopg is not installed.
        click.ClickException: If authentication fails and no fallback available.
    """
    try:
        from sqlalchemy.exc import OperationalError

        from olmo_eval.storage.backends.postgres.session import (
            get_database_session as _get_database_session,
        )
    except ImportError:
        console.print(
            "[red]Error:[/red] Database support requires psycopg. "
            "Install with: pip install psycopg[binary]"
        )
        raise SystemExit(1) from None

    # Apply defaults
    host = db_host or DEFAULT_DB_HOST
    port = db_port or DEFAULT_DB_PORT
    database = db_name or DEFAULT_DB_NAME
    user = db_user or DEFAULT_DB_USER
    password = db_password

    # First attempt with provided/env password
    try:
        return _get_database_session(host, port, database, user, password or "")
    except OperationalError as e:
        if not _is_auth_failure(e):
            raise

        # Auth failed - try fetching from AWS Secrets Manager
        arn = os.environ.get("OLMO_EVAL_DB_SECRET_ARN")
        if not arn:
            raise click.ClickException(
                "Database authentication failed. Set OLMO_EVAL_DB_PASSWORD or "
                "OLMO_EVAL_DB_SECRET_ARN environment variable, or use --db-password."
            ) from e

        logger.info("Authentication failed, retrying with AWS Secrets Manager")
        password = _fetch_password_from_aws(arn)

        # Retry with fetched password
        return _get_database_session(host, port, database, user, password)
