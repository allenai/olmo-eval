"""Google credentials for the ingest service.

The ingest service verifies a Google OAuth access token on every request. Both
``gcloud auth application-default login`` credentials and service-account keys
produce one. Inside Beaker, gantry writes the launching user's
``<beakeruser>_GOOGLE_CREDENTIALS`` secret to a file and points
GOOGLE_APPLICATION_CREDENTIALS at it.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
)
NO_CREDENTIALS_MESSAGE = (
    "No Google credentials found. Run `gcloud auth application-default login`, or pass --no-upload."
)
REFRESH_MARGIN = timedelta(minutes=5)


class UploadAuthError(Exception):
    """Google credentials are missing or cannot produce an access token."""


def load_google_credentials() -> Any:
    """Load Application Default Credentials with the scopes the ingest service needs.

    Raises:
        UploadAuthError: If no credentials are configured.
    """
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        credentials, _project = google.auth.default(scopes=list(SCOPES))
    except DefaultCredentialsError as e:
        raise UploadAuthError(NO_CREDENTIALS_MESSAGE) from e
    return credentials


def _expires_soon(expiry: datetime | None) -> bool:
    if expiry is None:
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    return expiry - datetime.now(UTC) < REFRESH_MARGIN


class GoogleTokenProvider:
    """Thread-safe source of Google access tokens.

    Credentials load lazily on the first call, and the token refreshes when it is
    missing or expires within five minutes.
    """

    def __init__(self, credentials: Any | None = None) -> None:
        self._credentials = credentials
        self._lock = threading.Lock()

    def token(self) -> str:
        """Return a valid access token.

        Raises:
            UploadAuthError: If credentials are missing or the refresh fails.
        """
        with self._lock:
            if self._credentials is None:
                self._credentials = load_google_credentials()
            creds = self._credentials
            if not creds.token or not creds.valid or _expires_soon(creds.expiry):
                self._refresh(creds)
            if not creds.token:
                raise UploadAuthError(
                    "Google credentials returned no access token. Run "
                    "`gcloud auth application-default login`, or pass --no-upload."
                )
            return creds.token

    @staticmethod
    def _refresh(creds: Any) -> None:
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request

        try:
            creds.refresh(Request())
        except GoogleAuthError as e:
            raise UploadAuthError(
                f"Could not refresh Google credentials ({e}). Run "
                "`gcloud auth application-default login`, or pass --no-upload."
            ) from e
