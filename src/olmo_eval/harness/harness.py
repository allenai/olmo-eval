"""Harness: A model provider configured with specific capabilities.

The Harness wraps an InferenceProvider and applies configuration to all requests.
It provides both single-turn (generate) and multi-turn (run) interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams

from .backends import Backend, get_backend
from .config import HarnessConfig
from .result import HarnessResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider


class Harness:
    """A model provider configured with specific capabilities.

    The Harness wraps an InferenceProvider and applies
    configuration to the requests. It provides both single-turn (generate/agenerate)
    and multi-turn (run) interfaces.

    For multi-turn execution with run(), a backend must be configured.
    """

    def __init__(self, config: HarnessConfig) -> None:
        """Initialize the Harness.

        Args:
            config: Configuration specifying provider, tools, system prompt, etc.
                The provider is created from config.provider.
        """
        self.config = config
        self._provider: InferenceProvider | None = None
        self._backend: Backend | None = None

    @property
    def provider(self) -> InferenceProvider:
        """Get or create the inference provider.

        The provider is lazily created from config.provider on first access.
        """
        if self._provider is None:
            self._provider = self.config.provider.create_provider()
        return self._provider

    @property
    def backend(self) -> Backend:
        """Get or create the backend.

        The backend is lazily created from config.backend on first access.

        Raises:
            RuntimeError: If no backend is configured.
        """
        if self._backend is None:
            if not self.config.backend:
                raise RuntimeError(
                    "No backend configured. Set config.backend to use run(). "
                    "For single-turn generation, use generate() or agenerate() instead."
                )
            self._backend = get_backend(self.config.backend)
        return self._backend

    @property
    def model_name(self) -> str:
        """Get the model name from the provider.

        Returns:
            Model name string.
        """
        return self.provider.model_name

    # ─────────────────────────────────────────────────────────
    # Single-turn interface (same as Provider, but with config)
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Single-turn generation with config injected.

        Applies the harness configuration (tools, system prompt) to each request
        and passes them to the provider.

        Args:
            requests: List of requests to process.
            sampling_params: Optional sampling parameters.

        Returns:
            List of output lists, one per request.
        """
        transformed = [self._apply_config(r) for r in requests]
        return self.provider.generate(transformed, sampling_params)

    def logprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Log probability computation.

        Args:
            requests: List of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return self.provider.logprobs(requests)

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async single-turn generation with config injected."""
        transformed = [self._apply_config(r) for r in requests]
        return await self.provider.agenerate(transformed, sampling_params)

    async def alogprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Async log probability computation."""
        return await self.provider.alogprobs(requests)

    # ─────────────────────────────────────────────────────────
    # Multi-turn interface (delegates to backend)
    # ─────────────────────────────────────────────────────────

    async def run(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> HarnessResult:
        """Multi-turn execution via configured backend.

        Runs an agent loop that:
        1. Sends the request to the model
        2. If the response has tool calls, executes them
        3. Appends results and continues until done or max_turns

        Args:
            request: Initial request to start the conversation.
            sampling_params: Optional sampling parameters.
            trace_metadata: Optional metadata for tracing (e.g., instance_id, task_id).

        Returns:
            HarnessResult with trajectory and final output.

        Raises:
            RuntimeError: If no backend is configured.
        """
        return await self.backend.run(
            self.provider, self.config, request, sampling_params, trace_metadata
        )

    async def cleanup(self) -> None:
        """Clean up resources held by the harness and its backend."""
        if self._backend is not None:
            await self._backend.cleanup()

    # ─────────────────────────────────────────────────────────
    # Config application (used by backends)
    # ─────────────────────────────────────────────────────────

    def _apply_config(self, request: LMRequest) -> LMRequest:
        """Inject tool schemas and system prompt from config.

        This transforms a request by adding:
        - Tool schemas (if config has tools)
        - System prompt (if configured and not already present)

        Args:
            request: Original request.

        Returns:
            New request with config applied.
        """
        messages = self._inject_system_prompt(request.messages)

        return LMRequest(
            request_type=request.request_type,
            messages=messages,
            prompt=request.prompt,
            continuations=request.continuations,
            tools=self.config.tool_schemas if self.config.has_tools else request.tools,
            system_prompt=self.config.system_prompt or request.system_prompt,
        )

    def _inject_system_prompt(
        self, messages: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        """Add system prompt to messages if configured and not present.

        Args:
            messages: Original message tuple.

        Returns:
            Messages with system prompt prepended if needed.
        """
        if not self.config.system_prompt:
            return messages

        # Check if messages already start with a system message
        if messages and messages[0].get("role") == "system":
            return messages

        # Prepend system message
        system_msg: dict[str, Any] = {
            "role": "system",
            "content": self.config.system_prompt,
        }
        return (system_msg,) + messages


def create_harness(
    config: HarnessConfig | dict[str, Any],
) -> Harness:
    """Create a Harness from config.

    Convenience function that handles config creation/parsing.

    Args:
        config: HarnessConfig instance or dict.

    Returns:
        Configured Harness instance.
    """
    if isinstance(config, dict):
        return Harness(HarnessConfig.from_dict(cast(dict[str, Any], config)))
    return Harness(config)
