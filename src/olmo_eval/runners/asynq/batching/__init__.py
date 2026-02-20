"""Batching strategies for async evaluation processing.

This module provides configurable batching strategies that control how
items are grouped and sent to the inference provider.

Strategies:
    - sequential: One batch at a time, wait for completion (default)
    - continuous: No batching, stream directly to provider (all-at-once)

Example:
    >>> from olmo_eval.runners.asynq.batching import BatchConfig, get_strategy
    >>> config = BatchConfig.sequential(chunk_size=256)
    >>> strategy = get_strategy(config)
"""

from .base import BatchingStrategy
from .config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_TIMEOUT,
    BatchConfig,
    BatchStrategy,
)
from .continuous import ContinuousStrategy
from .sequential import SequentialStrategy


def get_strategy(config: BatchConfig) -> BatchingStrategy:
    """Factory to get strategy implementation.

    Args:
        config: Batch configuration specifying strategy and parameters.

    Returns:
        Configured batching strategy instance.

    Raises:
        ValueError: If strategy is not recognized.
    """
    strategies = {
        BatchStrategy.SEQUENTIAL: SequentialStrategy,
        BatchStrategy.CONTINUOUS: ContinuousStrategy,
    }

    strategy_cls = strategies.get(config.strategy)
    if strategy_cls is None:
        raise ValueError(
            f"Unknown batch strategy: {config.strategy}. Available: {list(strategies.keys())}"
        )

    return strategy_cls(config)


__all__ = [
    # Config
    "BatchConfig",
    "BatchStrategy",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_TIMEOUT",
    # Strategies
    "BatchingStrategy",
    "SequentialStrategy",
    "ContinuousStrategy",
    # Factory
    "get_strategy",
]
