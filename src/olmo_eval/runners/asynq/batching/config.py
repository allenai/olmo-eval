"""Batch processing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar


class BatchStrategy(StrEnum):
    """Available batching strategies."""

    SEQUENTIAL = "sequential"  # One batch at a time, wait for completion
    PIPELINED = "pipelined"  # Multiple batches in flight concurrently
    CONTINUOUS = "continuous"  # No batching, stream directly to provider


# Default values
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_TIMEOUT = 5.0
DEFAULT_MAX_IN_FLIGHT = 2
DEFAULT_STAGGER_DELAY = 10.0  # seconds between batch starts during initial ramp


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        strategy: Batching strategy to use.
        chunk_size: Maximum items per batch.
        chunk_timeout: Seconds to wait for batch to fill before processing partial.
        max_in_flight: Maximum concurrent batches (pipelined strategy only).
        stagger_delay: Seconds between batch starts during initial ramp (pipelined only).
    """

    # Providers that only support sequential (LLM() is not thread-safe)
    _SEQUENTIAL_ONLY: ClassVar[frozenset[str]] = frozenset({"vllm", "hf"})

    strategy: BatchStrategy = BatchStrategy.SEQUENTIAL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_timeout: float = DEFAULT_CHUNK_TIMEOUT
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    stagger_delay: float = DEFAULT_STAGGER_DELAY

    def validate_for_provider(self, provider_kind: str) -> None:
        """Validate that this batching config is compatible with the provider.

        Args:
            provider_kind: The provider type (e.g., "vllm", "vllm_server").

        Raises:
            ValueError: If the batching strategy is not supported by the provider.
        """
        if provider_kind in self._SEQUENTIAL_ONLY and self.strategy != BatchStrategy.SEQUENTIAL:
            raise ValueError(
                f"Provider '{provider_kind}' only supports sequential batching. "
                f"Use vllm_server for {self.strategy} batching."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": str(self.strategy),
            "chunk_size": self.chunk_size,
            "chunk_timeout": self.chunk_timeout,
            "max_in_flight": self.max_in_flight,
            "stagger_delay": self.stagger_delay,
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
            stagger_delay=data.get("stagger_delay", DEFAULT_STAGGER_DELAY),
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
        stagger_delay: float = DEFAULT_STAGGER_DELAY,
    ) -> BatchConfig:
        """Create pipelined config for maximum GPU utilization."""
        return cls(
            strategy=BatchStrategy.PIPELINED,
            chunk_size=chunk_size,
            max_in_flight=max_in_flight,
            stagger_delay=stagger_delay,
        )

    @classmethod
    def continuous(cls) -> BatchConfig:
        """Create continuous config (no batching)."""
        return cls(strategy=BatchStrategy.CONTINUOUS)
