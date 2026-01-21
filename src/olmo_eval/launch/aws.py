"""AWS credential handling for Beaker jobs.

Provides utilities to retrieve local AWS credentials and store them as
user-scoped Beaker secrets for S3 access in evaluation jobs.

Example:
    from olmo_eval.launch.aws import ensure_aws_secrets, is_s3_path

    if is_s3_path(model_path):
        aws_secrets = ensure_aws_secrets(workspace="ai2/my-workspace")
        # Returns: [("AWS_ACCESS_KEY_ID", "username_AWS_ACCESS_KEY_ID"), ...]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beaker import Beaker

log = logging.getLogger(__name__)

__all__ = [
    "AWSCredentials",
    "get_local_aws_credentials",
    "is_s3_path",
    "ensure_aws_secrets",
]


@dataclass
class AWSCredentials:
    """AWS credentials for S3 access.

    Attributes:
        access_key_id: AWS access key ID.
        secret_access_key: AWS secret access key.
        session_token: Optional session token for temporary credentials.
    """

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


def get_local_aws_credentials() -> AWSCredentials | None:
    """Retrieve AWS credentials from the local environment.

    Uses boto3's credential chain which checks (in order):
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. Shared credentials file (~/.aws/credentials)
    3. AWS config file (~/.aws/config)
    4. IAM role (if running on AWS)

    Returns:
        AWSCredentials if found, None otherwise.
    """
    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed, cannot retrieve AWS credentials")
        return None

    session = boto3.Session()
    credentials = session.get_credentials()

    if credentials is None:
        return None

    # get_credentials() returns frozen credentials
    frozen = credentials.get_frozen_credentials()

    return AWSCredentials(
        access_key_id=frozen.access_key,
        secret_access_key=frozen.secret_key,
        session_token=frozen.token,  # May be None for long-term creds
    )


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 URL.

    Args:
        path: Path to check.

    Returns:
        True if the path starts with "s3://".
    """
    return path.startswith("s3://")


def _get_beaker_username(client: Beaker) -> str:
    """Get the current Beaker username.

    Args:
        client: Beaker client instance.

    Returns:
        The username of the authenticated Beaker account.
    """
    return client.user_name


def _write_secret_if_needed(
    client: Beaker,
    name: str,
    value: str,
    overwrite: bool,
) -> bool:
    """Write a secret to Beaker if it doesn't exist or overwrite is True.

    Args:
        client: Beaker client instance.
        name: Secret name.
        value: Secret value.
        overwrite: Whether to overwrite existing secrets.

    Returns:
        True if the secret was written, False if it already existed.
    """
    try:
        existing = client.secret.get(name)
        if existing and not overwrite:
            log.debug(f"Secret {name} already exists, skipping")
            return False
    except Exception:
        pass  # Secret doesn't exist

    client.secret.write(name, value)
    log.info(f"Wrote secret {name} to Beaker workspace")
    return True


def ensure_aws_secrets(
    workspace: str,
    credentials: AWSCredentials | None = None,
    overwrite: bool = False,
) -> list[tuple[str, str]]:
    """Ensure AWS credentials exist as user-scoped Beaker secrets.

    Secrets are stored with a username prefix to prevent collisions between
    users in shared workspaces. For example, user "alice" will have secrets
    named "alice_AWS_ACCESS_KEY_ID", "alice_AWS_SECRET_ACCESS_KEY", etc.

    The returned tuples map environment variable names to secret names,
    suitable for use with BeakerEnvSecret.

    Args:
        workspace: Beaker workspace to store secrets in.
        credentials: AWS credentials to store. If None, retrieves from local env.
        overwrite: Whether to overwrite existing secrets.

    Returns:
        List of (env_var_name, secret_name) tuples. For example:
        [("AWS_ACCESS_KEY_ID", "alice_AWS_ACCESS_KEY_ID"), ...]

    Raises:
        ValueError: If no credentials available.
    """
    from beaker import Beaker

    if credentials is None:
        credentials = get_local_aws_credentials()

    if credentials is None:
        raise ValueError(
            "No AWS credentials found. Please configure AWS credentials via:\n"
            "  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
            "  - AWS credentials file (~/.aws/credentials)\n"
            "  - AWS config file (~/.aws/config)"
        )

    client = Beaker.from_env(default_workspace=workspace)
    username = _get_beaker_username(client)
    secrets: list[tuple[str, str]] = []

    # User-scoped secret names
    access_key_secret = f"{username}_AWS_ACCESS_KEY_ID"
    secret_key_secret = f"{username}_AWS_SECRET_ACCESS_KEY"

    _write_secret_if_needed(client, access_key_secret, credentials.access_key_id, overwrite)
    secrets.append(("AWS_ACCESS_KEY_ID", access_key_secret))

    _write_secret_if_needed(client, secret_key_secret, credentials.secret_access_key, overwrite)
    secrets.append(("AWS_SECRET_ACCESS_KEY", secret_key_secret))

    # Store session token if present (for temporary/assumed role credentials)
    if credentials.session_token:
        session_token_secret = f"{username}_AWS_SESSION_TOKEN"
        _write_secret_if_needed(client, session_token_secret, credentials.session_token, overwrite)
        secrets.append(("AWS_SESSION_TOKEN", session_token_secret))

    return secrets
