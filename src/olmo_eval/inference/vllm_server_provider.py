"""vLLM Server provider for agent tasks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from olmo_eval.core.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.core.types.tools import ToolCall

from .base import InferenceProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from .vllm_server import VLLMServerProcess


class VLLMServerProvider(InferenceProvider):
    """Provider that uses a vLLM server's OpenAI-compatible API.

    This provider wraps a vLLM server URL (e.g., from vllm_server_context)
    and provides both the standard InferenceProvider interface and an
    AsyncOpenAI client for agent backends.

    Example:
        with vllm_server_context("meta-llama/Llama-3.1-8B-Instruct") as url:
            provider = VLLMServerProvider("Llama-3.1-8B-Instruct", base_url=url)
            harness = Harness(provider, config)
            result = await harness.run(request)
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_concurrency: int = 32,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier for requests.
            base_url: Base URL of the vLLM server. Defaults to "http://localhost:8000/v1".
            timeout: Request timeout in seconds.
            max_concurrency: Maximum number of concurrent requests.
        """
        super().__init__(model_name)
        self.base_url = base_url or "http://localhost:8000/v1"
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self._client: AsyncOpenAI | None = None

    def _get_or_create_client(self) -> AsyncOpenAI:
        """Get or create the AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def get_openai_client(self) -> AsyncOpenAI:
        """Get the AsyncOpenAI client for this provider."""
        return self._get_or_create_client()

    async def _generate_single_async(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        client = self._get_or_create_client()

        # Build messages
        if request.messages:
            messages: list[dict[str, Any]] = [dict(m) for m in request.messages]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # Build tools if present
        tools = None
        if request.tools:
            tools = [t.to_openai() for t in request.tools]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "n": params.num_samples,
            "max_tokens": params.max_tokens,
        }

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature
        if params.stop_sequences:
            # OpenAI API supports max 4 stop sequences
            kwargs["stop"] = list(params.stop_sequences)[:4]
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)

        outputs = []
        for choice in response.choices:
            text = choice.message.content or ""
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall.create(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in choice.message.tool_calls
                ]
            outputs.append(LMOutput(text=text, tool_calls=tool_calls))

        return outputs

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions via the vLLM server.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        params = self._default_sampling_params(sampling_params)

        async def arun() -> list[list[LMOutput]]:
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def process(req: LMRequest) -> list[LMOutput]:
                async with semaphore:
                    return await self._generate_single_async(req, params)

            return await asyncio.gather(*[process(r) for r in requests])

        return asyncio.run(arun())

    def logprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Compute logprobs (limited support via API).

        Note: The OpenAI-compatible API has limited logprobs support compared
        to direct model access.

        Raises:
            NotImplementedError: Logprobs are not fully supported via API.
        """
        raise NotImplementedError("Logprobs not fully supported via API")

    @classmethod
    def from_server_context(
        cls,
        model_name: str,
        server: VLLMServerProcess,
        **kwargs: Any,
    ) -> VLLMServerProvider:
        """Create a provider from a VLLMServerProcess.

        Convenience factory for use with vllm_server_context.

        Args:
            model_name: Model name for requests.
            server: VLLMServerProcess instance.
            **kwargs: Additional provider arguments.

        Returns:
            Configured VLLMServerProvider.

        Example:
            server = VLLMServerProcess("meta-llama/Llama-3.1-8B-Instruct")
            with server:
                provider = VLLMServerProvider.from_server_context(
                    "Llama-3.1-8B-Instruct", server
                )
                harness = Harness(provider, config)
        """
        return cls(model_name, base_url=server.base_url, **kwargs)
