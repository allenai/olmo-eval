"""Pluggable execution backends for Harness.

Backends are registered using the @register_backend decorator and define
how the harness executes requests.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from olmo_eval.common.types import LMRequest, SamplingParams
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)


class Backend:
    """Base class for harness execution backends.

    Backends define how the harness executes requests. Backends may optionally
    implement `run` for multi-turn execution support.

    Class Attributes:
        name: Human-readable name for this backend.
        required_extras: Tuple of pyproject.toml extra names required by this backend.
    """

    name: str = "base"
    required_extras: tuple[str, ...] = ()

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Execute the request and return the result.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration (tools, system prompt, etc.).
            request: The initial request to process.
            sampling_params: Optional sampling parameters override.

        Returns:
            HarnessResult with trajectory and final output.

        Raises:
            NotImplementedError: If this backend doesn't support run().
        """
        raise NotImplementedError(f"Backend '{self.name}' does not support run()")


# -----------------------------------------------------------------------------
# Backend Registry
# -----------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type[Backend]] = {}

T = TypeVar("T", bound=Backend)


def register_backend(name: str):
    """Decorator to register a Backend class.

    Usage:
        @register_backend("default")
        class DefaultBackend(Backend):
            ...

    Args:
        name: Name to register the backend under.

    Returns:
        Decorator function that registers the class.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in BACKEND_REGISTRY:
            logger.warning(f"Overwriting existing backend: {name}")
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str) -> Backend:
    """Get a backend instance by name.

    Args:
        name: Backend name (e.g., "default", "openai_agents").

    Returns:
        Backend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")
    return BACKEND_REGISTRY[name]()


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns:
        Sorted list of backend names.
    """
    return sorted(BACKEND_REGISTRY.keys())


def get_backend_extras(name: str) -> tuple[str, ...]:
    """Get the required extras for a backend by name.

    Args:
        name: Backend name (e.g., "default", "openai_agents").

    Returns:
        Tuple of pyproject.toml extra names required by the backend.

    Raises:
        ValueError: If backend name is unknown.
    """
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")
    return BACKEND_REGISTRY[name].required_extras


# Import backends to trigger registration
from .openai_agents import OpenAIAgentsBackend  # noqa: E402

__all__ = [
    "Backend",
    "BACKEND_REGISTRY",
    "OpenAIAgentsBackend",
    "get_backend",
    "get_backend_extras",
    "list_backends",
    "register_backend",
]
