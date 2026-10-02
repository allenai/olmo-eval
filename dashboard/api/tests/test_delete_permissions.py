"""Who may delete a run, and the uploader account allowlist default."""

from __future__ import annotations

import pytest

from olmo_eval_api.auth.google_token import Principal
from olmo_eval_api.services.ingest import can_delete
from olmo_eval_api.settings import UPLOADER_SERVICE_ACCOUNT, Settings

ALICE = Principal(email="alice@allenai.org", principal_type="user", exp=0)
UPLOADER = Principal(email=UPLOADER_SERVICE_ACCOUNT, principal_type="service_account", exp=0)


@pytest.mark.parametrize(
    ("principal", "uploaded_by", "author", "allowed"),
    [
        (ALICE, "alice@allenai.org", "alice", True),
        (ALICE, "bob@allenai.org", "alice", False),
        (ALICE, UPLOADER_SERVICE_ACCOUNT, "alice", True),
        (ALICE, UPLOADER_SERVICE_ACCOUNT, "bob", False),
        (ALICE, UPLOADER_SERVICE_ACCOUNT, None, False),
        (UPLOADER, UPLOADER_SERVICE_ACCOUNT, "alice", False),
        (UPLOADER, "alice@allenai.org", "alice", False),
    ],
)
def test_can_delete(
    principal: Principal, uploaded_by: str, author: str | None, allowed: bool
) -> None:
    assert can_delete(principal, uploaded_by, author) is allowed


def test_uploader_is_allowlisted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_ALLOWED_SERVICE_ACCOUNTS", raising=False)
    assert Settings().ingest_allowed_service_accounts == [UPLOADER_SERVICE_ACCOUNT]
