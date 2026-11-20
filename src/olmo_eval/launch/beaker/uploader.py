"""Uploader service account key for Beaker jobs.

Beaker jobs upload results to the dashboard's ingest service as the shared
``olmo-eval-uploader`` service account. The account has no Google Cloud roles; the
ingest service allowlists it. Its key lives in Secret Manager (managed by
infra/terraform/uploader.tf), and any @allenai.org account can read it.

At launch, the key is read with the launching user's local credentials and written
to a shared Beaker secret in the job's workspace, so personal credentials never
reach Beaker. The secret is rewritten when the key rotates. A launch whose local
credentials cannot read the key reuses the workspace's existing secret, so launches
from machines without a current Google login still upload. Inside the job, the CLI
moves the key out of the environment before any subprocess starts.
"""

from __future__ import annotations

import base64
import logging

log = logging.getLogger(__name__)

UPLOADER_PROJECT = "ai2-skiff2-olmo-eval"
UPLOADER_KEY_SECRET = "olmo-eval-uploader-key"
UPLOADER_SECRET_NAME = "olmo_eval_uploader_key"

_SECRET_URL = (
    f"https://secretmanager.googleapis.com/v1/projects/{UPLOADER_PROJECT}"
    f"/secrets/{UPLOADER_KEY_SECRET}/versions/latest:access"
)


class UploaderKeyError(Exception):
    """The uploader key could not be read from Secret Manager."""


def fetch_uploader_key() -> str:
    """Read the uploader key JSON from Secret Manager with local credentials.

    Raises:
        UploaderKeyError: If credentials are missing or access is denied.
    """
    import google.auth
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import AuthorizedSession

    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        response = AuthorizedSession(credentials).get(_SECRET_URL, timeout=30)
    except GoogleAuthError as e:
        raise UploaderKeyError(
            f"Could not read the uploader key ({e}). Run "
            "`gcloud auth application-default login`, or pass --no-upload."
        ) from e
    if response.status_code != 200:
        raise UploaderKeyError(
            f"Could not read secret {UPLOADER_KEY_SECRET} in {UPLOADER_PROJECT} "
            f"(HTTP {response.status_code}: {response.text[:300]}). Check that "
            "`gcloud auth application-default login` used your @allenai.org account, "
            "or pass --no-upload."
        )
    return base64.b64decode(response.json()["payload"]["data"]).decode()


def uploader_secret_exists(workspace: str) -> bool:
    """Whether the workspace already holds the uploader secret."""
    from beaker import Beaker
    from beaker.exceptions import BeakerSecretNotFound

    try:
        Beaker.from_env(default_workspace=workspace).secret.get(UPLOADER_SECRET_NAME)
    except BeakerSecretNotFound:
        return False
    return True


def ensure_uploader_secret(workspace: str, key: str | None = None) -> str:
    """Make sure the workspace's uploader secret holds the current key.

    When the key cannot be read from Secret Manager (no local Google credentials, or
    they need reauthentication) but the workspace already holds the secret, the
    existing secret is used as is. A secret older than a key rotation then fails at
    upload time inside the job, which logs the error and keeps the results.

    Args:
        workspace: Beaker workspace the job runs in.
        key: Uploader key JSON. Read from Secret Manager when None.

    Returns:
        The Beaker secret name holding the key.

    Raises:
        UploaderKeyError: If the key cannot be read and the workspace has no secret.
    """
    from beaker import Beaker

    if key is None:
        try:
            key = fetch_uploader_key()
        except UploaderKeyError as e:
            if not uploader_secret_exists(workspace):
                raise
            log.warning(
                f"{e} Using the existing Beaker secret {UPLOADER_SECRET_NAME} in "
                f"{workspace} instead."
            )
            return UPLOADER_SECRET_NAME
    client = Beaker.from_env(default_workspace=workspace)
    try:
        current = client.secret.read(client.secret.get(UPLOADER_SECRET_NAME))
    except Exception:
        current = None
    if current != key:
        client.secret.write(UPLOADER_SECRET_NAME, key)
        log.info(f"Wrote uploader key to Beaker secret {UPLOADER_SECRET_NAME} in {workspace}")
    return UPLOADER_SECRET_NAME
