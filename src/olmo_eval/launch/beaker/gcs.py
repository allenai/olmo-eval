"""Google credential handling for Beaker jobs.

Provides utilities to retrieve local Google credentials and store them as
user-scoped Beaker secrets. Jobs use them for GCS access and to upload results
to the dashboard as the launching user.

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


SUPPORTED_CREDENTIAL_TYPES = ("service_account", "authorized_user")


@dataclass
class GCSCredentials:
    """Google credentials JSON for Beaker jobs.

    Attributes:
        json_key: The full JSON content of the credentials file.
        project_id: The GCP project ID, when the file records one.
        client_email: The service account email (service-account keys only).
        credential_type: "service_account" or "authorized_user".
    """

    json_key: str
    project_id: str | None = None
    client_email: str | None = None
    credential_type: str = "service_account"


def _read_credentials(path: Path) -> GCSCredentials | None:
    try:
        json_key = path.read_text()
        data = json.loads(json_key)
    except Exception as e:
        log.warning(f"Could not read {path}: {e}")
        return None
    cred_type = data.get("type") if isinstance(data, dict) else None
    if cred_type not in SUPPORTED_CREDENTIAL_TYPES:
        log.warning(
            f"Found {path} but its type is {cred_type!r}; expected a service account key or "
            "`gcloud auth application-default login` credentials."
        )
        return None
    log.debug(f"Found Google {cred_type} credentials at {path}")
    return GCSCredentials(
        json_key=json_key,
        project_id=data.get("project_id") or data.get("quota_project_id"),
        client_email=data.get("client_email"),
        credential_type=cred_type,
    )


def get_local_gcs_credentials() -> GCSCredentials | None:
    """Retrieve Google credentials from the local environment.

    Checks (in order):
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable (path to a JSON file)
    2. gcloud application default credentials
       (~/.config/gcloud/application_default_credentials.json)

    Service-account keys and user credentials from
    ``gcloud auth application-default login`` are both accepted. Beaker jobs use
    them for GCS access and to upload results to the dashboard as the launching user.

    Returns:
        GCSCredentials if found, None otherwise.
    """
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        path = Path(creds_path).expanduser()
        if path.exists():
            creds = _read_credentials(path)
            if creds is not None:
                return creds

    default_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if default_path.exists():
        return _read_credentials(default_path)

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
    """Write a secret unless it already holds the same value (or overwrite is False).

    Args:
        client: Beaker client instance.
        name: Secret name.
        value: Secret value.
        overwrite: Write even when a secret with this name exists. When False, an
            existing secret is still replaced if its value differs from ``value``.

    Returns:
        True if the secret was written, False if it already held this value.
    """
    if not overwrite:
        try:
            existing = client.secret.get(name)
        except Exception:
            existing = None  # Secret doesn't exist
        if existing:
            try:
                current = client.secret.read(existing)
            except Exception:
                current = None
            if current == value:
                log.debug(f"Secret {name} is up to date, skipping")
                return False

    client.secret.write(name, value)
    log.info(f"Wrote secret {name} to Beaker workspace")
    return True


def ensure_gcs_secrets(
    workspace: str,
    credentials: GCSCredentials | None = None,
    overwrite: bool = False,
) -> str:
    """Ensure Google credentials exist as a user-scoped Beaker secret.

    The secret is stored with a username prefix to prevent collisions between
    users in shared workspaces. For example, user "alice" will have a secret
    named "alice_GOOGLE_CREDENTIALS".

    Unlike AWS credentials which use multiple env vars, Google credentials are
    stored as a single secret containing the full credentials JSON (service-account
    key or authorized-user ADC). This is what gantry's google_credentials_secret
    parameter expects. An existing secret is replaced when its value differs from
    the local credentials, so refreshed ADC reaches new jobs.

    Args:
        workspace: Beaker workspace to store secrets in.
        credentials: Credentials to store. If None, retrieves from local env.
        overwrite: Write the secret even when it already holds the same value.

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
            "No Google credentials found. Run `gcloud auth application-default login`, "
            "or set GOOGLE_APPLICATION_CREDENTIALS to a service account key file."
        )

    client = Beaker.from_env(default_workspace=workspace)
    username = _get_beaker_username(client)

    # User-scoped secret name
    secret_name = f"{username}_GOOGLE_CREDENTIALS"

    _write_secret_if_needed(client, secret_name, credentials.json_key, overwrite)

    return secret_name
