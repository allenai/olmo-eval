"""Common secret handling for Beaker jobs.

Provides utilities to retrieve local secrets (HuggingFace token, Weights & Biases key)
and store them as user-scoped Beaker secrets.

Example:
    from olmo_eval.launch.beaker.secrets import ensure_common_secrets

    # Ensure HF_TOKEN and WANDB_API_KEY exist in Beaker
    common_secrets = ensure_common_secrets(workspace="ai2/my-workspace")
    # Returns: [("HF_TOKEN", "username_HF_TOKEN"), ("WANDB_API_KEY", "username_WANDB_API_KEY")]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from olmo_eval.launch.beaker.constants import (
    OLMO_EVAL_DB_ARN_SECRET_NAME,
    OLMO_EVAL_PGHOST_SECRET_NAME,
    STORE_DEFAULTS,
)

if TYPE_CHECKING:
    from beaker import Beaker

log = logging.getLogger(__name__)

__all__ = [
    "COMMON_SECRET_NAMES",
    "get_local_hf_token",
    "get_local_wandb_api_key",
    "ensure_common_secrets",
    "ensure_task_secrets",
    "get_aws_secret_value",
    "get_store_secret_mappings",
    "get_store_env_defaults",
    "setup_pgpassword_from_arn",
]

COMMON_SECRET_NAMES: frozenset[str] = frozenset({"HF_TOKEN"})


def get_local_hf_token() -> str | None:
    """Retrieve HuggingFace token from the local environment.

    Checks (in order):
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
    3. ~/.huggingface/token file (huggingface-cli login)
    4. ~/.cache/huggingface/token file (older location)

    Returns:
        HuggingFace token if found, None otherwise.
    """
    # Check environment variables first
    token = os.environ.get("HF_TOKEN")
    if token:
        log.debug("Found HF_TOKEN in environment")
        return token

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        log.debug("Found HUGGING_FACE_HUB_TOKEN in environment")
        return token

    # Check token files
    token_paths = [
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ]

    for token_path in token_paths:
        if token_path.exists():
            try:
                token = token_path.read_text().strip()
                if token:
                    log.debug(f"Found HF token in {token_path}")
                    return token
            except Exception as e:
                log.warning(f"Could not read {token_path}: {e}")

    return None


def get_local_wandb_api_key() -> str | None:
    """Retrieve Weights & Biases API key from the local environment.

    Checks (in order):
    1. WANDB_API_KEY environment variable
    2. ~/.netrc file (wandb login)

    Returns:
        WANDB API key if found, None otherwise.
    """
    # Check environment variable first
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        log.debug("Found WANDB_API_KEY in environment")
        return api_key

    # Check netrc file
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            import netrc

            nrc = netrc.netrc(str(netrc_path))
            auth = nrc.authenticators("api.wandb.ai")
            if auth:
                # netrc returns (login, account, password) - API key is the password
                api_key = auth[2]
                if api_key:
                    log.debug("Found WANDB API key in ~/.netrc")
                    return api_key
        except Exception as e:
            log.warning(f"Could not read ~/.netrc for wandb credentials: {e}")

    return None


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


def ensure_common_secrets(
    workspace: str,
    overwrite: bool = False,
) -> list[tuple[str, str]]:
    """Ensure common secrets (HF_TOKEN, WANDB_API_KEY) exist as user-scoped Beaker secrets.

    Secrets are stored with a username prefix to prevent collisions between
    users in shared workspaces. For example, user "alice" will have secrets
    named "alice_HF_TOKEN", "alice_WANDB_API_KEY".

    The returned tuples map environment variable names to secret names,
    suitable for use with BeakerEnvSecret.

    Unlike ensure_aws_secrets, this function does NOT raise an error if
    credentials are not found - it simply skips that secret and logs a warning.
    This allows jobs to run even without all optional credentials.

    Args:
        workspace: Beaker workspace to store secrets in.
        overwrite: Whether to overwrite existing secrets.

    Returns:
        List of (env_var_name, secret_name) tuples for secrets that were
        found and written. For example:
        [("HF_TOKEN", "alice_HF_TOKEN"), ("WANDB_API_KEY", "alice_WANDB_API_KEY")]
    """
    from beaker import Beaker

    client = Beaker.from_env(default_workspace=workspace)
    username = _get_beaker_username(client)
    secrets: list[tuple[str, str]] = []

    # Handle HF_TOKEN
    hf_token = get_local_hf_token()
    if hf_token:
        hf_secret_name = f"{username}_HF_TOKEN"
        _write_secret_if_needed(client, hf_secret_name, hf_token, overwrite)
        secrets.append(("HF_TOKEN", hf_secret_name))
    else:
        log.warning(
            "No HuggingFace token found. Set HF_TOKEN environment variable "
            "or run 'huggingface-cli login' to enable authenticated HF access."
        )

    # Handle WANDB_API_KEY
    wandb_key = get_local_wandb_api_key()
    if wandb_key:
        wandb_secret_name = f"{username}_WANDB_API_KEY"
        _write_secret_if_needed(client, wandb_secret_name, wandb_key, overwrite)
        secrets.append(("WANDB_API_KEY", wandb_secret_name))
    else:
        log.warning(
            "No Weights & Biases API key found. Set WANDB_API_KEY environment variable "
            "or run 'wandb login' to enable W&B logging."
        )

    return secrets


def ensure_task_secrets(
    workspace: str,
    required_secrets: set[str],
) -> list[tuple[str, str]]:
    """Ensure task-required secrets exist in Beaker.

    Unlike ensure_common_secrets, this function DOES raise an error if
    any required secret is not found. Task-required secrets are mandatory
    for the evaluation to run correctly.

    Secrets are expected to be stored with a username prefix to prevent
    collisions between users in shared workspaces. For example, user "alice"
    requesting "S2_API_KEY" will look for secret "alice_S2_API_KEY".

    Args:
        workspace: Beaker workspace to check secrets in.
        required_secrets: Set of environment variable names that must exist
            as Beaker secrets.

    Returns:
        List of (env_var_name, secret_name) tuples.

    Raises:
        ValueError: If any required secret is not found in Beaker.
    """
    if not required_secrets:
        return []

    from beaker import Beaker
    from beaker.exceptions import BeakerSecretNotFound

    client = Beaker.from_env(default_workspace=workspace)
    username = _get_beaker_username(client)
    secrets: list[tuple[str, str]] = []
    missing: list[str] = []

    for env_var in sorted(required_secrets):
        secret_name = f"{username}_{env_var}"
        try:
            client.secret.get(secret_name)
            secrets.append((env_var, secret_name))
            log.debug(f"Found required secret {secret_name}")
        except BeakerSecretNotFound:
            missing.append(f"{env_var} (expected Beaker secret: {secret_name})")

    if missing:
        raise ValueError(
            "Missing required Beaker secrets:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nCreate these secrets with:\n"
            + "\n".join(f"  beaker secret write {username}_{s.split()[0]} <value>" for s in missing)
        )

    return secrets


def get_aws_secret_value(secret_arn: str, key: str | None = None) -> str:
    """Fetch a secret value from AWS Secrets Manager.

    Args:
        secret_arn: The ARN of the secret.
        key: If provided, parse secret as JSON and extract this key.

    Returns:
        The secret value.
    """
    import boto3

    # Extract region from ARN (format: arn:aws:secretsmanager:REGION:ACCOUNT:secret:NAME)
    parts = secret_arn.split(":")
    region = parts[3] if len(parts) >= 4 else "us-east-1"

    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    secret_string = response["SecretString"]

    if key is None:
        return secret_string

    return json.loads(secret_string)[key]


def get_store_secret_mappings() -> list[tuple[str, str]]:
    """Get environment variable to Beaker secret mappings for --store.

    Returns:
        List of (env_var_name, secret_name) tuples.
    """
    return [
        ("DB_SECRET_ARN", OLMO_EVAL_DB_ARN_SECRET_NAME),
        ("PGHOST", OLMO_EVAL_PGHOST_SECRET_NAME),
    ]


def get_store_env_defaults() -> dict[str, str]:
    """Get default environment variables for --store.

    These can be overridden by the user in the launch command.

    Returns:
        Dict of env_var_name -> default_value.
    """
    return STORE_DEFAULTS.copy()


PGPASSWORD_FILE = "/tmp/.pgpassword"


def setup_pgpassword_from_arn() -> None:
    """Fetch PGPASSWORD from AWS Secrets Manager and write to secure file.

    Writes to /tmp/.pgpassword with mode 600. The shell should source this
    file and then delete it.

    Raises:
        SystemExit: If DB_SECRET_ARN is not set or password fetch fails.
    """
    import sys

    arn = os.environ.get("DB_SECRET_ARN")
    if arn is None:
        print("Error: DB_SECRET_ARN environment variable not set", file=sys.stderr)
        sys.exit(1)
        return  # unreachable, helps type checker

    try:
        password = get_aws_secret_value(arn, key="password")
        pgpass_file = Path(PGPASSWORD_FILE)
        pgpass_file.write_text(f"export PGPASSWORD='{password}'\n")
        pgpass_file.chmod(0o600)
    except Exception as e:
        print(
            f"Error: Failed to fetch password from {OLMO_EVAL_DB_ARN_SECRET_NAME}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
