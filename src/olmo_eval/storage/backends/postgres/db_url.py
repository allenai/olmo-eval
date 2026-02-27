"""Database URL builder with AWS Secrets Manager support."""

from __future__ import annotations

import logging
import os
from urllib.parse import quote_plus

log = logging.getLogger(__name__)


def get_password_from_aws() -> str | None:
    """Fetch database password from AWS Secrets Manager.

    Uses OLMO_EVAL_DB_SECRET_ARN environment variable.

    Returns:
        Password string if successful, None if ARN not set.

    Raises:
        Exception: If ARN is set but fetch fails.
    """
    arn = os.environ.get("OLMO_EVAL_DB_SECRET_ARN")
    if not arn:
        return None

    from olmo_eval.launch.beaker.secrets import get_aws_secret_value

    log.debug(f"Fetching password from AWS Secrets Manager: {arn}")
    return get_aws_secret_value(arn, key="password")


def build_database_url(database: str) -> str | None:
    """Build a PostgreSQL database URL from environment variables.

    Uses OLMO_EVAL_DB_* environment variables (same as CLI).

    Args:
        database: Database name.

    Returns:
        Database URL string, or None if host is not set.
    """
    host = os.environ.get("OLMO_EVAL_DB_HOST")
    if not host:
        return None

    port = os.environ.get("OLMO_EVAL_DB_PORT", "5432")
    user = os.environ.get("OLMO_EVAL_DB_USER", "postgres")

    password = os.environ.get("OLMO_EVAL_DB_PASSWORD")
    if not password:
        password = get_password_from_aws()
    if not password:
        password = "postgres"

    password_encoded = quote_plus(password)
    return f"postgresql+psycopg://{user}:{password_encoded}@{host}:{port}/{database}"


def build_results_db_url() -> str | None:
    """Build URL for the results database (olmo_eval)."""
    database = os.environ.get("OLMO_EVAL_DB_NAME", "olmo_eval")
    url = build_database_url(database=database)
    if url:
        log.info(f"Built results database URL for {database}")
    return url


def build_metrics_db_url() -> str | None:
    """Build URL for the metrics database (olmo_eval_metrics)."""
    url = build_database_url(database="olmo_eval_metrics")
    if url:
        log.info("Built metrics database URL")
    return url
