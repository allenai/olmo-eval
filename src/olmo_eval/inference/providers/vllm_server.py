"""vLLM Server provider for agent tasks."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import httpx

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, SamplingParams
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.retry import retry_with_backoff

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from .vllm_server_utils import VLLMServerProcess

logger = get_logger(__name__)

# Enable with VLLM_DEBUG_REQUESTS=1
_DEBUG_REQUESTS = os.environ.get("VLLM_DEBUG_REQUESTS", "").lower() in ("1", "true", "yes")


def _log_request(request: httpx.Request) -> None:
    """Log outgoing HTTP request."""
    body = request.content.decode("utf-8", errors="replace") if request.content else ""
    # Truncate very long bodies
    if len(body) > 2000:
        body = body[:2000] + "... [truncated]"
    logger.info(f"vLLM request: {request.method} {request.url}\n  body: {body}")


class VLLMServerProvider(InferenceProvider):
    """Provider that uses a vLLM server's OpenAI-compatible API.

    This provider can either connect to an existing vLLM server (if base_url
    is provided) or start and manage its own server subprocess.

    Example:
        # Auto-start server (managed lifecycle)
        provider = VLLMServerProvider("meta-llama/Llama-3.1-8B-Instruct")

        # Or connect to existing server
        provider = VLLMServerProvider("model", base_url="http://localhost:8000/v1")
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_concurrency: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        # Server config (only used if base_url is None)
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        tokenizer: str | None = None,
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str | None = None,
        trust_remote_code: bool = False,
        log_dir: str | None = None,
        **server_kwargs: Any,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier for requests.
            base_url: Base URL of existing vLLM server. If None, starts own server.
            timeout: Request timeout in seconds.
            max_concurrency: Maximum number of concurrent requests.
            max_retries: Maximum number of retries for transient errors.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            tensor_parallel_size: Number of GPUs for tensor parallelism (server mode).
            max_model_len: Maximum model context length (server mode).
            tokenizer: Tokenizer path override (server mode).
            enable_auto_tool_choice: Enable automatic tool choice (server mode).
            tool_call_parser: Tool call parser name (server mode).
            trust_remote_code: Trust remote code for model loading (server mode).
            log_dir: Directory to write server logs to (server mode).
            **server_kwargs: Additional vLLM server arguments.
        """
        super().__init__(model_name)
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._openai_module: Any = None
        self._server: VLLMServerProcess | None = None  # type: ignore[name-defined]

        if base_url:
            # Connect to existing server
            self.base_url = base_url
        else:
            # Start our own server
            from .vllm_server_utils import VLLMServerProcess

            # Build server kwargs
            srv_kwargs: dict[str, Any] = dict(server_kwargs)
            if max_model_len:
                srv_kwargs["max_model_len"] = max_model_len
            if tokenizer:
                srv_kwargs["tokenizer"] = tokenizer
            if enable_auto_tool_choice:
                srv_kwargs["enable_auto_tool_choice"] = True
                if tool_call_parser:
                    srv_kwargs["tool_call_parser"] = tool_call_parser
            if trust_remote_code:
                srv_kwargs["trust_remote_code"] = True

            self._server = VLLMServerProcess(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                log_dir=log_dir,
                **srv_kwargs,
            )
            self._server.start()
            self.base_url = self._server.base_url

    def close(self) -> None:
        """Close the provider and stop managed server if any."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def __del__(self) -> None:
        """Ensure server is stopped on garbage collection."""
        self.close()

    def _get_or_create_client(self) -> AsyncOpenAI:
        """Get or create the AsyncOpenAI client."""
        if self._client is None:
            import openai
            from openai import AsyncOpenAI

            self._openai_module = openai

            # Configure connection pool limits to prevent exhaustion with large batches.
            # Default httpx settings (100 connections) can cause issues when processing
            # thousands of instances with agent loops making multiple API calls each.
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0,  # Close idle connections after 30s
            )

            # Build event hooks for debug logging if enabled
            event_hooks: dict[str, list[Any]] | None = None
            if _DEBUG_REQUESTS:
                logger.info("vLLM debug request logging enabled (VLLM_DEBUG_REQUESTS=1)")
                event_hooks = {
                    "request": [_log_request],
                }

            self._http_client = httpx.AsyncClient(
                limits=limits,
                timeout=self.timeout,
                event_hooks=event_hooks or {},
            )

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # Disable SDK retries - we handle them ourselves
                http_client=self._http_client,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the provider and release resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    def get_openai_client(self) -> AsyncOpenAI:
        """Get the AsyncOpenAI client for this provider."""
        return self._get_or_create_client()

    async def _generate_single_impl(
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

    async def _generate_single_async(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        return await retry_with_backoff(
            lambda: self._generate_single_impl(request, params),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            context=f"generate model={self.model_name}",
            sdk_module=self._openai_module,
        )

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions via the vLLM server.

        This is the native async implementation that should be used
        in async contexts to avoid nested event loops.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        params = self._default_sampling_params(sampling_params)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process(req: LMRequest) -> list[LMOutput]:
            async with semaphore:
                return await self._generate_single_async(req, params)

        return await asyncio.gather(*[process(r) for r in requests])

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions via the vLLM server (sync wrapper).

        For async contexts, prefer using agenerate() directly.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        return asyncio.run(self.agenerate(requests, sampling_params))

    async def _logprobs_single_impl(self, request: LMRequest) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        client = self._get_or_create_client()

        if request.messages:
            content = request.messages[0].get("content", "") if request.messages else ""
        else:
            content = request.prompt

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=50,
            temperature=0.0,
            logprobs=True,
        )

        completion_logprobs: list[LogProbEntry] = []
        if response.choices:
            choice = response.choices[0]
            logprobs_data = getattr(choice, "logprobs", None)
            if logprobs_data and hasattr(logprobs_data, "content") and logprobs_data.content:
                for lp in logprobs_data.content:
                    entry: LogProbEntry = {"token": lp.token, "logprob": lp.logprob}
                    lp_bytes = getattr(lp, "bytes", None)
                    if lp_bytes is not None:
                        entry["bytes"] = lp_bytes
                    completion_logprobs.append(entry)

        outputs = []
        for continuation in request.continuations or ():
            total = (
                sum(lp["logprob"] for lp in completion_logprobs[:5]) if completion_logprobs else 0.0
            )
            outputs.append(
                LMOutput(
                    text=continuation,
                    logprobs=completion_logprobs[:5] if completion_logprobs else None,
                    metadata={"total_logprob": total},
                )
            )

        return outputs

    async def _logprobs_single_async(self, request: LMRequest) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        return await retry_with_backoff(
            lambda: self._logprobs_single_impl(request),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            context=f"logprobs model={self.model_name}",
            sdk_module=self._openai_module,
        )

    async def alogprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process(req: LMRequest) -> list[LMOutput]:
            async with semaphore:
                return await self._logprobs_single_async(req)

        return await asyncio.gather(*[process(r) for r in requests])

    def logprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Compute logprobs for continuations.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return asyncio.run(self.alogprobs(requests))
