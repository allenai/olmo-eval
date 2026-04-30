"""Push progress updates to the current Beaker workload's description.

When code runs inside a Beaker job, ``BEAKER_WORKLOAD_ID`` is set in the
environment. This module wraps that detail and provides a small reporter
that pushes throttled status messages to the workload description so they
appear in the Beaker UI while the job is running.

Outside of a Beaker job (env var unset) the reporter is a no-op.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beaker import Beaker, BeakerWorkload

logger = logging.getLogger(__name__)

_BEAKER_AVAILABLE = importlib.util.find_spec("beaker") is not None

DEFAULT_MIN_INTERVAL = 10.0


def _beaker_credentials_present() -> bool:
    if os.environ.get("BEAKER_TOKEN"):
        return True
    config_path = os.environ.get("BEAKER_CONFIG") or os.path.expanduser("~/.beaker/config.yml")
    return os.path.exists(config_path)


def _git_suffix() -> str:
    commit = os.environ.get("GIT_COMMIT") or os.environ.get("GIT_REF") or "unknown"
    branch = os.environ.get("GIT_BRANCH") or "unknown"
    return f"git_commit: {commit} git_branch: {branch}"


class BeakerStatusReporter:
    """Throttled writer for the current Beaker workload's description."""

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL) -> None:
        self.min_interval = min_interval
        self._workload_id = os.environ.get("BEAKER_WORKLOAD_ID") or os.environ.get(
            "BEAKER_EXPERIMENT_ID"
        )
        has_creds = _beaker_credentials_present()
        self.enabled = bool(self._workload_id) and _BEAKER_AVAILABLE and has_creds
        if self._workload_id and not self.enabled:
            missing = (
                "beaker-py"
                if not _BEAKER_AVAILABLE
                else "credentials (BEAKER_TOKEN or ~/.beaker/config.yml)"
            )
            logger.warning(
                "BEAKER_WORKLOAD_ID set but %s missing; Beaker status updates disabled.",
                missing,
            )
        self._git_suffix = _git_suffix()
        self._client: Beaker | None = None
        self._workload: BeakerWorkload | None = None
        self._last_update: float = float("-inf")
        self._last_message: str | None = None

    def _ensure_client(self) -> bool:
        if not self.enabled or self._workload_id is None:
            return False
        if self._client is not None and self._workload is not None:
            return True
        from beaker import Beaker

        self._client = Beaker.from_env()
        self._workload = self._client.workload.get(self._workload_id)
        return True

    def update(self, message: str, force: bool = False) -> None:
        """Push a status message to the Beaker workload description.

        Throttled by ``min_interval`` so callers can call this on every loop
        iteration. No-op when not running inside a Beaker job.
        """
        if not self.enabled:
            return

        now = time.monotonic()
        if not force and now - self._last_update < self.min_interval:
            return
        if message == self._last_message and not force:
            return

        if not self._ensure_client():
            return

        full_message = f"{message} {self._git_suffix}"
        assert self._client is not None and self._workload is not None
        self._client.workload.update(self._workload, description=full_message)
        self._last_update = now
        self._last_message = message

    def report_progress(
        self,
        label: str,
        count: int,
        total: int,
        start_time: float,
        units: str = "items/sec",
        force: bool = False,
    ) -> None:
        """Format and push a standard progress message.

        ``start_time`` must be a ``time.monotonic()`` value.
        """
        if not self.enabled:
            return
        elapsed = max(time.monotonic() - start_time, 1e-9)
        rate = count / elapsed
        pct = (count / total * 100) if total > 0 else 0.0
        self.update(
            f"{label} {count}/{total} ({pct:.0f}%) at {rate:.4f} {units}",
            force=force,
        )

    def progress_callback(self, label: str, units: str = "items/sec") -> Callable[..., None]:
        """Return a ``(count, total, *, force=False)`` callback bound to a fresh start time.

        When called without ``force``, forces a flush iff ``count == total``.
        Suitable for passing as ``on_progress`` to ``dispatch_concurrent``.
        """
        start = time.monotonic()

        def _cb(count: int, total: int, *, force: bool = False) -> None:
            self.report_progress(
                label, count, total, start, units=units, force=force or count == total
            )

        return _cb
