"""Dashboard identity from IAP headers (spec 4.1).

IAP authenticates every request to the dashboard service and sets
``X-Goog-Authenticated-User-Email: accounts.google.com:user@allenai.org``. The header is trusted
only in dashboard mode, where IAP is in front of the container.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from olmo_eval_api.errors import ApiError
from olmo_eval_api.log_config import principal_var
from olmo_eval_api.settings import Settings

IAP_EMAIL_HEADER = "x-goog-authenticated-user-email"


@dataclass(frozen=True)
class DashboardUser:
    email: str
    dev_mode: bool

    @property
    def username(self) -> str:
        return self.email.split("@", 1)[0]


async def current_user(request: Request) -> DashboardUser:
    settings: Settings = request.app.state.settings
    raw = request.headers.get(IAP_EMAIL_HEADER)
    if raw:
        email = raw.split(":", 1)[1] if ":" in raw else raw
        user = DashboardUser(email=email.strip().lower(), dev_mode=False)
    elif settings.is_local:
        user = DashboardUser(email=settings.dashboard_dev_user.lower(), dev_mode=True)
    else:
        raise ApiError(401, "missing IAP identity")
    principal_var.set(user.email)
    return user
