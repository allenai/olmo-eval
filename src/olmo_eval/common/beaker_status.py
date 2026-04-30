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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beaker import Beaker, BeakerWorkload

logger = logging.getLogger(__name__)

_BEAKER_AVAILABLE = importlib.util.find_spec("beaker") is not None

DEFAULT_MIN_INTERVAL = 10.0


def _git_suffix() -> str:
    """Return ``git_commit: X git_branch: Y`` suffix from env vars (or unknown)."""
    commit = os.environ.get("GIT_COMMIT") or "unknown"
    branch = os.environ.get("GIT_BRANCH") or "unknown"
    return f"git_commit: {commit} git_branch: {branch}"


class BeakerStatusReporter:
    """Throttled writer for the current Beaker workload's description."""

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL) -> None:
        """Initialize the reporter.

        Args:
            min_interval: Minimum seconds between non-forced updates.
        """
        self.min_interval = min_interval
        self._workload_id = os.environ.get("BEAKER_WORKLOAD_ID") or os.environ.get(
            "BEAKER_EXPERIMENT_ID"
        )
        self.enabled = bool(self._workload_id) and _BEAKER_AVAILABLE
        if bool(self._workload_id) and not _BEAKER_AVAILABLE:
            logger.warning(
                "BEAKER_WORKLOAD_ID set but beaker-py not installed; "
                "Beaker status updates disabled."
            )
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

        Args:
            message: One-line status message.
            force: If True, bypass the interval throttle.
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

        full_message = f"{message} {_git_suffix()}"
        assert self._client is not None and self._workload is not None
        self._client.workload.update(self._workload, description=full_message)
        self._last_update = now
        self._last_message = message
