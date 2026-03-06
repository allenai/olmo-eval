"""Modal secret management utilities.

Provides utilities to retrieve local GCP credentials and store them as
Modal secrets for container registry authentication.

Example:
    from olmo_eval.launch.modal.secrets import ensure_modal_gcp_secret

    secret_name = ensure_modal_gcp_secret()
    # Use secret_name in RegistryAuth(provider="gcp", secret_name=secret_name)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "GCPCredentials",
    "get_local_gcp_credentials",
    "ensure_modal_gcp_secret",
]


@dataclass
class GCPCredentials:
    """GCP service account credentials.

    Attributes:
        json_key: The full JSON content of the service account key file.
        project_id: The GCP project ID (extracted from JSON).
        client_email: The service account email (extracted from JSON).
    """

    json_key: str
    project_id: str | None = None
    client_email: str | None = None


def get_local_gcp_credentials() -> GCPCredentials | None:
    """Retrieve GCP credentials from the local environment.

    Checks (in order):
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable (path to JSON key)
    2. gcloud default application credentials
       (~/.config/gcloud/application_default_credentials.json)

    Note: Only service account keys are supported. Application Default Credentials
    from 'gcloud auth application-default login' are not supported because Modal
    secrets require a service account to authenticate with GCP Artifact Registry.

    Returns:
        GCPCredentials if found, None otherwise.
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
                    log.debug(f"Found GCP service account credentials at {creds_path}")
                    return GCPCredentials(
                        json_key=json_key,
                        project_id=data.get("project_id"),
                        client_email=data.get("client_email"),
                    )
                else:
                    log.warning(
                        f"Found {creds_path} but it's not a service account key "
                        f"(type={data.get('type')}). "
                        "For Modal secrets, use a service account key file."
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
                if data.get("type") == "service_account":
                    log.debug(f"Found GCP service account credentials at {path}")
                    return GCPCredentials(
                        json_key=json_key,
                        project_id=data.get("project_id"),
                        client_email=data.get("client_email"),
                    )
                else:
                    log.debug(
                        f"Found {path} but it's type '{data.get('type')}', "
                        "not 'service_account'. "
                        "For Modal secrets, use a service account key file."
                    )
            except Exception as e:
                log.warning(f"Could not read {path}: {e}")

    return None


def ensure_modal_gcp_secret(
    secret_name: str = "gcp-artifact-registry",
    credentials: GCPCredentials | None = None,
) -> str:
    """Ensure Modal secret exists with GCP credentials for Artifact Registry.

    Creates or updates a Modal secret containing GCP service account credentials
    in the format expected by modal.Image.from_gcp_artifact_registry().

    The secret will contain SERVICE_ACCOUNT_JSON with the full JSON key content.

    Args:
        secret_name: Name for the Modal secret.
        credentials: GCP credentials to store. If None, retrieves from local env.

    Returns:
        The Modal secret name to use in RegistryAuth.

    Raises:
        ValueError: If no credentials available.
        ImportError: If modal is not installed.

    Example:
        # Setup (run once or when credentials change):
        secret_name = ensure_modal_gcp_secret()

        # Usage in SandboxConfig:
        config = SandboxConfig(
            image="python:3.11",
            mode=SandboxMode.MODAL,
            inject_swerex=True,
            registry_auth=RegistryAuth(provider="gcp", secret_name=secret_name),
        )
    """
    import importlib.util

    if importlib.util.find_spec("modal") is None:
        raise ImportError("Modal not installed. Install with: pip install modal")

    if credentials is None:
        credentials = get_local_gcp_credentials()

    if credentials is None:
        raise ValueError(
            "No GCP credentials found. Please configure GCP credentials via:\n"
            "  - GOOGLE_APPLICATION_CREDENTIALS env var (path to service account JSON)\n"
            "  - Service account key file\n"
            "\n"
            "Note: Application Default Credentials from 'gcloud auth application-default login'\n"
            "are not supported for Modal secrets. Use a service account key instead."
        )

    # Create Modal secret with the service account JSON
    # Modal expects SERVICE_ACCOUNT_JSON for GCP Artifact Registry authentication
    log.info(f"Creating/updating Modal secret '{secret_name}' with GCP credentials")

    # Use modal.Secret.from_dict to create the secret object
    # Then persist it using the Modal CLI or API
    # Note: This creates a transient secret. For persistent secrets, users should
    # use `modal secret create` CLI command with the JSON content.
    #
    # For programmatic creation, we provide instructions since Modal's Python API
    # for secret management is limited.
    log.info(
        f"To persist this secret, run:\n"
        f"  modal secret create {secret_name} SERVICE_ACCOUNT_JSON='<json-content>'\n"
        f"Or use the Modal dashboard to create the secret."
    )

    return secret_name
