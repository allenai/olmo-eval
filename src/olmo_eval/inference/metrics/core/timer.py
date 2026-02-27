"""Timer utilities for measuring latency."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class Timer:
    """Context manager for timing operations."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        if self._end > 0:
            return self._end - self._start
        # Still running
        return time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_s * 1000


@dataclass
class TokenTimestamps:
    """Track timestamps for token-level timing.

    Used to compute time-to-first-token and per-token latency when
    streaming is available.
    """

    start_time: float = 0.0
    first_token_time: float | None = None
    token_times: list[float] = field(default_factory=list)

    def start(self) -> None:
        """Mark the start of generation."""
        self.start_time = time.perf_counter()

    def record_token(self) -> None:
        """Record timestamp for a token."""
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)

    @property
    def time_to_first_token_s(self) -> float | None:
        """Time to first token in seconds."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time

    @property
    def mean_time_per_token_s(self) -> float | None:
        """Mean time per output token in seconds."""
        if len(self.token_times) < 2:
            return None
        # Compute mean inter-token time
        deltas = [
            self.token_times[i] - self.token_times[i - 1] for i in range(1, len(self.token_times))
        ]
        return sum(deltas) / len(deltas)
