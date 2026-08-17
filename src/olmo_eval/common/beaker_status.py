"""Push progress updates to the current Beaker workload's description.

When code runs inside a Beaker job, ``BEAKER_WORKLOAD_ID`` is set in the
environment. This module wraps that detail and provides a small reporter
that pushes throttled status messages to the workload description so they
appear in the Beaker UI while the job is running.

Outside of a Beaker job (env var unset) the reporter is a no-op.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

from beaker import Beaker, BeakerWorkload
from beaker.exceptions import BeakerConfigurationError

DEFAULT_MIN_INTERVAL = 10.0

logger = logging.getLogger(__name__)


def _git_suffix() -> str:
    commit = os.environ.get("GIT_COMMIT") or os.environ.get("GIT_REF") or "unknown"
    branch = os.environ.get("GIT_BRANCH") or "unknown"
    return f"git_commit: {commit} git_branch: {branch}"


class BeakerStatusReporter:
    """Throttled writer for the current Beaker workload's description."""

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL) -> None:
        self.min_interval = min_interval
        self._git_suffix = _git_suffix()
        self._workload: BeakerWorkload | None = None
        self._last_update: float = float("-inf")
        try:
            self._client: Beaker | None = Beaker.from_env()
        except BeakerConfigurationError:
            self._client = None
            return
        self._workload = self._client.workload.get(os.environ["BEAKER_WORKLOAD_ID"])

    def update(self, message: str, force: bool = False) -> None:
        """Push a status message to the Beaker workload description.

        Throttled by ``min_interval`` so callers can call this on every loop
        iteration. No-op when not running inside a Beaker job.
        """
        if self._client is None or self._workload is None:
            return

        now = time.monotonic()
        if not force and now - self._last_update < self.min_interval:
            return

        full_message = f"{message} {self._git_suffix}"
        self._client.workload.update(self._workload, description=full_message)
        self._last_update = now

    def progress_callback(self, label: str, units: str = "items/sec") -> Callable[..., None]:
        """Return a ``(count, total, *, force=False)`` callback bound to a fresh start time.

        When called without ``force``, forces a flush iff ``count == total``.
        Suitable for passing as ``on_progress`` to ``dispatch_concurrent``.
        """
        start = time.monotonic()

        def _cb(count: int, total: int, *, force: bool = False) -> None:
            if self._client is None:
                return
            try:
                elapsed = max(time.monotonic() - start, 1e-9)
                rate = count / elapsed
                pct = (count / total * 100) if total > 0 else 0.0
                self.update(
                    f"{label} {count}/{total} ({pct:.0f}%) at {rate:.4f} {units}",
                    force=force or count == total,
                )
            except Exception:
                logger.exception("BeakerStatusReporter progress callback failed")

        return _cb
