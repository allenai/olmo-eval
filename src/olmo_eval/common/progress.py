"""Progress logging utilities for non-TTY environments."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger

DEFAULT_LOG_INTERVAL = 30.0  # seconds


class ProgressLogger:
    """Time-based progress logger that works without a TTY.

    Logs progress at regular intervals showing count, percentage, and rate.
    Suitable for batch processing in server/CI environments where tqdm
    doesn't render properly.

    Example output:
        Processing: 45/164 (27%) at 1.0 items/sec
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        logger: Logger | None = None,
        log_interval: float = DEFAULT_LOG_INTERVAL,
    ) -> None:
        """Initialize progress logger.

        Args:
            total: Total number of items to process.
            desc: Description prefix for log messages.
            logger: Logger to use (defaults to module logger).
            log_interval: Seconds between progress logs.
        """
        self.total = total
        self.desc = desc
        self.logger = logger or logging.getLogger(__name__)
        self.log_interval = log_interval

        self.count = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(self, n: int = 1) -> None:
        """Update progress count and log if interval elapsed.

        Args:
            n: Number of items completed (default 1).
        """
        self.count += n
        now = time.time()

        if now - self.last_log_time >= self.log_interval:
            self._log_progress(now)
            self.last_log_time = now

    def _log_progress(self, now: float) -> None:
        """Log current progress."""
        elapsed = now - self.start_time
        rate = self.count / elapsed if elapsed > 0 else 0.0
        pct = (self.count / self.total * 100) if self.total > 0 else 0.0

        self.logger.info(
            f"{self.desc}: {self.count}/{self.total} ({pct:.0f}%) at {rate:.1f} items/sec"
        )

    def close(self) -> None:
        """Log final progress summary."""
        elapsed = time.time() - self.start_time
        rate = self.count / elapsed if elapsed > 0 else 0.0

        self.logger.info(
            f"{self.desc}: done {self.count}/{self.total} in {elapsed:.1f}s ({rate:.1f} items/sec)"
        )

    def __enter__(self) -> ProgressLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
