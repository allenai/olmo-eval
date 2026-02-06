"""Harness: A model configured with specific capabilities.

The Harness wraps an InferenceProvider and applies configuration to all requests.
It provides both single-turn (generate) and multi-turn (run) interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmo_eval.core.types import LMOutput, LMRequest, SamplingParams

from .backend import get_backend
from .config import HarnessConfig
from .result import HarnessResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider


class Harness:
    """A model configured with specific capabilities.

    Wraps an InferenceProvider and applies configuration to all requests.
    Uses pluggable backends for execution.

    The Harness provides two main interfaces:
    - generate(): Single-turn generation with config injected (tools, system prompt)
    - run(): Multi-turn execution with automatic tool handling

    Example:
        from olmo_eval.core.harness import Harness, HarnessConfig
        from olmo_eval.inference import VLLMProvider

        # Create provider
        provider = VLLMProvider("llama3.1-8b")

        # Create harness with tools
        config = HarnessConfig(
            name="search",
            tool_names=("web_search", "fetch_page"),
            system_prompt="You have access to search tools.",
        )
        harness = Harness(provider, config)

        # Single-turn generation
        outputs = harness.generate([request])

        # Multi-turn execution
        result = await harness.run(request)
    """

    def __init__(self, provider: InferenceProvider, config: HarnessConfig) -> None:
        """Initialize the Harness.

        Args:
            provider: The inference provider for model calls.
            config: Configuration specifying tools, system prompt, etc.
        """
        self.provider = provider
        self.config = config
        self.backend = get_backend(config.backend)

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

        Note: Config injection is optional for logprobs since tools
        typically aren't relevant for perplexity-style evaluation.

        Args:
            requests: List of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return self.provider.logprobs(requests)

    # ─────────────────────────────────────────────────────────
    # Multi-turn interface (delegates to backend)
    # ─────────────────────────────────────────────────────────

    async def run(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Multi-turn execution via configured backend.

        Runs an agent loop that:
        1. Sends the request to the model
        2. If the response has tool calls, executes them
        3. Appends results and continues until done or max_turns

        Args:
            request: Initial request to start the conversation.
            sampling_params: Optional sampling parameters.

        Returns:
            HarnessResult with trajectory and final output.
        """
        return await self.backend.run(self, request, sampling_params)

    async def run_batch(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[HarnessResult]:
        """Execute multiple multi-turn requests with concurrency control.

        Args:
            requests: List of initial requests.
            sampling_params: Optional sampling parameters.

        Returns:
            List of HarnessResult, one per request.
        """
        import asyncio

        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def run_one(request: LMRequest) -> HarnessResult:
            async with semaphore:
                return await self.run(request, sampling_params)

        return await asyncio.gather(*[run_one(r) for r in requests])

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
    provider: InferenceProvider,
    config: HarnessConfig | dict[str, Any] | None = None,
) -> Harness:
    """Create a Harness from provider and optional config.

    Convenience function that handles config creation/parsing.

    Args:
        provider: The inference provider.
        config: HarnessConfig instance, dict, or None for default.

    Returns:
        Configured Harness instance.
    """
    resolved: HarnessConfig
    if config is None:
        resolved = HarnessConfig(name="default")
    elif isinstance(config, dict):
        resolved = HarnessConfig.from_dict(config)  # type: ignore[arg-type]
    else:
        resolved = config

    return Harness(provider, resolved)
