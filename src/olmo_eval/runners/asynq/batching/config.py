"""Batch processing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BatchStrategy(StrEnum):
    """Available batching strategies."""

    SEQUENTIAL = "sequential"  # One batch at a time, wait for completion
    PIPELINED = "pipelined"  # Multiple batches in flight concurrently
    CONTINUOUS = "continuous"  # No batching, stream directly to provider


# Default values
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_TIMEOUT = 30.0
DEFAULT_MAX_IN_FLIGHT = 2


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        strategy: Batching strategy to use.
        chunk_size: Maximum items per batch.
        chunk_timeout: Seconds to wait for batch to fill before processing partial.
        max_in_flight: Maximum concurrent batches (pipelined strategy only).
    """

    strategy: BatchStrategy = BatchStrategy.SEQUENTIAL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_timeout: float = DEFAULT_CHUNK_TIMEOUT
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": str(self.strategy),
            "chunk_size": self.chunk_size,
            "chunk_timeout": self.chunk_timeout,
            "max_in_flight": self.max_in_flight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchConfig:
        """Create from dictionary."""
        strategy = data.get("strategy", BatchStrategy.SEQUENTIAL)
        if isinstance(strategy, str):
            strategy = BatchStrategy(strategy)
        return cls(
            strategy=strategy,
            chunk_size=data.get("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_timeout=data.get("chunk_timeout", DEFAULT_CHUNK_TIMEOUT),
            max_in_flight=data.get("max_in_flight", DEFAULT_MAX_IN_FLIGHT),
        )

    @classmethod
    def sequential(cls, chunk_size: int = DEFAULT_CHUNK_SIZE) -> BatchConfig:
        """Create sequential config."""
        return cls(strategy=BatchStrategy.SEQUENTIAL, chunk_size=chunk_size)

    @classmethod
    def pipelined(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    ) -> BatchConfig:
        """Create pipelined config for maximum GPU utilization."""
        return cls(
            strategy=BatchStrategy.PIPELINED,
            chunk_size=chunk_size,
            max_in_flight=max_in_flight,
        )

    @classmethod
    def continuous(cls) -> BatchConfig:
        """Create continuous config (no batching)."""
        return cls(strategy=BatchStrategy.CONTINUOUS)
