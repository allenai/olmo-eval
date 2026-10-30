"""Batch processing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Literal


class BatchStrategy(StrEnum):
    """Available batching strategies."""

    BATCHED = "batched"  # Process in chunks with continuous dispatch
    STREAMING = "streaming"  # Stream items directly to provider


# Default values
DEFAULT_CHUNK_SIZE = 64
DEFAULT_CHUNK_TIMEOUT = 5.0


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        strategy: Batching strategy to use.
        chunk_size: Maximum items per batch.
        chunk_timeout: Seconds to wait for batch to fill before processing partial.
        max_batch_sequences: Optional budget counting expanded samples, retained on replay.
    """

    # Providers that only support sequential (LLM() is not thread-safe)
    _SEQUENTIAL_ONLY: ClassVar[frozenset[str]] = frozenset(
        {"sglang", "vllm", "hf", "olmo_core", "olmo_core_vlm"}
    )

    strategy: BatchStrategy = BatchStrategy.BATCHED
    chunk_size: int | Literal["auto"] = "auto"
    chunk_timeout: float = DEFAULT_CHUNK_TIMEOUT
    max_batch_sequences: int | None = None

    @property
    def resolved_chunk_size(self) -> int:
        """Automatic batching keeps the historical size unless a provider recommends more."""
        return DEFAULT_CHUNK_SIZE if self.chunk_size == "auto" else self.chunk_size

    def __post_init__(self) -> None:
        if self.max_batch_sequences is not None and (
            type(self.max_batch_sequences) is not int or self.max_batch_sequences < 1
        ):
            raise ValueError("max_batch_sequences must be a positive integer or None")
        if self.chunk_size != "auto" and (type(self.chunk_size) is not int or self.chunk_size < 1):
            raise ValueError("chunk_size must be 'auto' or a positive integer")

    def validate_for_provider(self, provider_kind: str) -> None:
        """Validate that this batching config is compatible with the provider.

        Args:
            provider_kind: The provider type (e.g., "vllm", "vllm_server").

        Raises:
            ValueError: If the batching strategy is not supported by the provider.
        """
        if provider_kind in self._SEQUENTIAL_ONLY and self.strategy != BatchStrategy.BATCHED:
            raise ValueError(
                f"Provider '{provider_kind}' only supports batched processing. "
                f"Use vllm_server for streaming."
            )

    @classmethod
    def dispatches_per_request(cls, provider_kind: str) -> bool:
        """Whether the provider sends a batch's requests one by one instead of batching.

        Engine-in-process providers (``LLM``-style) batch a call's requests inside the
        engine, so splitting a batch would cost throughput. The others dispatch each
        request on its own, so a batch's requests can be sent as separate calls.
        """
        return provider_kind not in cls._SEQUENTIAL_ONLY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data: dict[str, Any] = {
            "strategy": str(self.strategy),
            "chunk_size": self.chunk_size,
            "chunk_timeout": self.chunk_timeout,
        }
        if self.max_batch_sequences is not None:
            data["max_batch_sequences"] = self.max_batch_sequences
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchConfig:
        """Create from dictionary."""
        strategy = data.get("strategy", BatchStrategy.BATCHED)
        if isinstance(strategy, str):
            strategy = BatchStrategy(strategy)
        return cls(
            strategy=strategy,
            chunk_size=data.get("chunk_size", "auto"),
            chunk_timeout=data.get("chunk_timeout", DEFAULT_CHUNK_TIMEOUT),
            max_batch_sequences=data.get("max_batch_sequences"),
        )

    @classmethod
    def batched(cls, chunk_size: int | Literal["auto"] = "auto") -> BatchConfig:
        """Create batched config with continuous dispatch within each chunk."""
        return cls(strategy=BatchStrategy.BATCHED, chunk_size=chunk_size)

    @classmethod
    def streaming(cls) -> BatchConfig:
        """Create streaming config (items sent directly to provider)."""
        return cls(strategy=BatchStrategy.STREAMING)
