"""GCS credential handling for Beaker jobs.

Provides utilities to retrieve local GCS credentials and store them as
user-scoped Beaker secrets for GCS access in evaluation jobs.

Example:
    from olmo_eval.launch.beaker.gcs import ensure_gcs_secrets, is_gcs_path

    if is_gcs_path(model_path):
        gcs_secret = ensure_gcs_secrets(workspace="ai2/my-workspace")
        # Returns: "username_GOOGLE_CREDENTIALS"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beaker import Beaker

log = logging.getLogger(__name__)

__all__ = [
    "GCSCredentials",
    "get_local_gcs_credentials",
    "is_gcs_path",
    "ensure_gcs_secrets",
]


@dataclass
class GCSCredentials:
    """GCS service account credentials.

    Attributes:
        json_key: The full JSON content of the service account key file.
        project_id: The GCP project ID (extracted from JSON).
        client_email: The service account email (extracted from JSON).
    """

    json_key: str
    project_id: str | None = None
    client_email: str | None = None


def get_local_gcs_credentials() -> GCSCredentials | None:
    """Retrieve GCS credentials from the local environment.

    Checks (in order):
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable (path to JSON key)
    2. gcloud default application credentials
       (~/.config/gcloud/application_default_credentials.json)

    Note: Only service account keys are supported. Application Default Credentials
    from 'gcloud auth application-default login' are not supported because Beaker
    jobs require a service account to authenticate.

    Returns:
        GCSCredentials if found, None otherwise.
    """
    # Check GOOGLE_APPLICATION_CREDENTIALS first
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        path = Path(creds_path).expanduser()
        if path.exists():
            try:
                json_key = path.read_text()
                data = json.loads(json_key)
                if data.get("type") == "service_account":
                    log.debug(f"Found GCS service account credentials at {creds_path}")
                    return GCSCredentials(
                        json_key=json_key,
                        project_id=data.get("project_id"),
                        client_email=data.get("client_email"),
                    )
                else:
                    log.warning(
                        f"Found {creds_path} but it's not a service account key "
                        f"(type={data.get('type')}). "
                        "For Beaker jobs, use a service account key file."
                    )
            except Exception as e:
                log.warning(f"Could not read {creds_path}: {e}")

    # Check gcloud default application credentials
    default_paths = [
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
    ]

    for path in default_paths:
        if path.exists():
            try:
                json_key = path.read_text()
                data = json.loads(json_key)
                # application_default_credentials.json may have different types
                if data.get("type") == "service_account":
                    log.debug(f"Found GCS service account credentials at {path}")
                    return GCSCredentials(
                        json_key=json_key,
                        project_id=data.get("project_id"),
                        client_email=data.get("client_email"),
                    )
                else:
                    log.debug(
                        f"Found {path} but it's type '{data.get('type')}', not 'service_account'. "
                        "For Beaker jobs, use a service account key file."
                    )
            except Exception as e:
                log.warning(f"Could not read {path}: {e}")

    return None


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS URL.

    Args:
        path: Path to check.

    Returns:
        True if the path starts with "gs://".
    """
    return path.startswith("gs://")


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


def ensure_gcs_secrets(
    workspace: str,
    credentials: GCSCredentials | None = None,
    overwrite: bool = False,
) -> str:
    """Ensure GCS credentials exist as a user-scoped Beaker secret.

    The secret is stored with a username prefix to prevent collisions between
    users in shared workspaces. For example, user "alice" will have a secret
    named "alice_GOOGLE_CREDENTIALS".

    Unlike AWS credentials which use multiple env vars, GCS credentials are
    stored as a single secret containing the full service account JSON key.
    This is what gantry's google_credentials_secret parameter expects.

    Args:
        workspace: Beaker workspace to store secrets in.
        credentials: GCS credentials to store. If None, retrieves from local env.
        overwrite: Whether to overwrite existing secrets.

    Returns:
        The Beaker secret name containing the GCS credentials JSON.

    Raises:
        ValueError: If no credentials available.
    """
    from beaker import Beaker

    if credentials is None:
        credentials = get_local_gcs_credentials()

    if credentials is None:
        raise ValueError(
            "No GCS credentials found. Please configure GCS credentials via:\n"
            "  - GOOGLE_APPLICATION_CREDENTIALS env var (path to service account JSON)\n"
            "  - Service account key file\n"
            "\n"
            "Note: Application Default Credentials from 'gcloud auth application-default login'\n"
            "are not supported for Beaker jobs. Use a service account key instead."
        )

    client = Beaker.from_env(default_workspace=workspace)
    username = _get_beaker_username(client)

    # User-scoped secret name
    secret_name = f"{username}_GOOGLE_CREDENTIALS"

    _write_secret_if_needed(client, secret_name, credentials.json_key, overwrite)

    return secret_name
