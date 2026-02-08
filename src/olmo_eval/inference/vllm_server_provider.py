"""vLLM Server provider for agent tasks."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

import httpx

from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.core.types.tools import ToolCall

from .base import InferenceProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from .vllm_server import VLLMServerProcess

# HTTP status codes that should trigger a retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# OpenAI exception type names that should never be retried
_NEVER_RETRY_TYPES = (
    "AuthenticationError",  # 401 – bad API key
    "BadRequestError",  # 400 – invalid params, content policy, etc.
    "NotFoundError",  # 404 – wrong model / endpoint
    "UnprocessableEntityError",  # 422 – semantic validation failure
)

# OpenAI exception type names that are always transient and should be retried
_ALWAYS_RETRY_TYPES = (
    "RateLimitError",  # 429
    "APITimeoutError",  # request timed out
    "APIConnectionError",  # connection-level failure
    "InternalServerError",  # 500
)

logger = get_logger(__name__)

# Enable with VLLM_DEBUG_REQUESTS=1
_DEBUG_REQUESTS = os.environ.get("VLLM_DEBUG_REQUESTS", "").lower() in ("1", "true", "yes")

T = TypeVar("T")


def _log_request(request: httpx.Request) -> None:
    """Log outgoing HTTP request."""
    body = request.content.decode("utf-8", errors="replace") if request.content else ""
    # Truncate very long bodies
    if len(body) > 2000:
        body = body[:2000] + "... [truncated]"
    logger.info(f"vLLM request: {request.method} {request.url}\n  body: {body}")


async def _log_response(response: httpx.Response) -> None:
    """Log incoming HTTP response."""
    # Read the response body (needed for streaming responses)
    await response.aread()
    body = response.text
    # Truncate very long bodies
    if len(body) > 2000:
        body = body[:2000] + "... [truncated]"
    logger.info(f"vLLM response: {response.status_code} {response.reason_phrase}\n  body: {body}")


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
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier for requests.
            base_url: Base URL of the vLLM server. Defaults to "http://localhost:8000/v1".
            timeout: Request timeout in seconds.
            max_concurrency: Maximum number of concurrent requests.
            max_retries: Maximum number of retries for transient errors.
            retry_delay: Base delay in seconds between retries (exponential backoff).
        """
        super().__init__(model_name)
        self.base_url = base_url or "http://localhost:8000/v1"
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None  # For debug logging
        self._openai_module: Any = None  # Cached openai module for exception types

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
                    "response": [_log_response],
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

    def _is_retryable(self, exc: Exception) -> bool:
        """Determine whether *exc* should be retried.

        Uses OpenAI's typed exception hierarchy to classify errors:
        - Never retry: AuthenticationError, BadRequestError, NotFoundError,
          UnprocessableEntityError
        - Always retry: RateLimitError, APITimeoutError, APIConnectionError,
          InternalServerError
        - Falls back to HTTP status code for unknown subtypes.
        """
        if self._openai_module is None:
            # Module not imported yet, fall back to status code check
            status = getattr(exc, "status_code", None)
            return status is not None and int(status) in _RETRYABLE_STATUS_CODES

        # Never retry these – the request itself is wrong
        for attr in _NEVER_RETRY_TYPES:
            cls = getattr(self._openai_module, attr, None)
            if cls is not None and isinstance(exc, cls):
                return False

        # Always retry these – transient server/network issues
        for attr in _ALWAYS_RETRY_TYPES:
            cls = getattr(self._openai_module, attr, None)
            if cls is not None and isinstance(exc, cls):
                return True

        # Fall back to HTTP status code for any openai error subtypes
        # not explicitly listed above; unknown / non-openai exceptions are not retried
        status = getattr(exc, "status_code", None)
        return status is not None and int(status) in _RETRYABLE_STATUS_CODES

    @staticmethod
    def _format_error(exc: Exception) -> str:
        """Build a detailed, single-log-entry description of *exc*."""
        parts: list[str] = [f"  type: {type(exc).__qualname__}"]

        status = getattr(exc, "status_code", None)
        if status is not None:
            parts.append(f"  status_code: {status}")

        # Include response body if available (OpenAI SDK stores this)
        response = getattr(exc, "response", None)
        if response is not None:
            parts.append(f"  url: {response.url}")

        message = getattr(exc, "message", None) or str(exc)
        # Truncate very long messages (e.g. full HTML error pages)
        if len(message) > 500:
            message = message[:500] + "…"
        parts.append(f"  message: {message}")

        # The wrapped cause often has the real reason (e.g. httpx.ReadTimeout)
        cause = exc.__cause__
        if cause is not None:
            parts.append(f"  cause: {type(cause).__qualname__}: {cause}")

        return "\n".join(parts)

    async def _retry_with_backoff_async(
        self, func: Callable[[], Awaitable[T]], *, context: str = ""
    ) -> T:
        """Execute with exponential backoff for retryable errors.

        Args:
            func: Async callable to execute.
            context: Optional human-readable label (e.g. ``"generate model=llama3"``)
                     included in log messages.

        Returns:
            Result of the function call.

        Raises:
            Exception: If all retries are exhausted or a non-retryable error occurs.
        """
        last_exception: Exception | None = None
        ctx = f" [{context}]" if context else ""

        for attempt in range(self.max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                detail = self._format_error(e)

                # Authentication errors: fail immediately with actionable guidance
                if self._openai_module is not None:
                    auth_cls = getattr(self._openai_module, "AuthenticationError", None)
                    if auth_cls is not None and isinstance(e, auth_cls):
                        logger.error(
                            f"Authentication failed{ctx}:\n{detail}\n"
                            f"  Verify the API key environment variable is set correctly."
                        )
                        raise

                    # Not-found errors: fail immediately with actionable guidance
                    not_found_cls = getattr(self._openai_module, "NotFoundError", None)
                    if not_found_cls is not None and isinstance(e, not_found_cls):
                        logger.error(
                            f"Resource not found{ctx}:\n{detail}\n"
                            f"  Verify the model name and API endpoint are correct."
                        )
                        raise

                retryable = self._is_retryable(e)

                if not retryable or attempt >= self.max_retries:
                    if retryable:
                        logger.error(
                            f"Retries exhausted{ctx} after {attempt + 1} attempts:\n{detail}"
                        )
                    else:
                        logger.error(f"Non-retryable error{ctx}:\n{detail}")
                    raise

                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Retryable error{ctx} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}):\n{detail}\n"
                    f"  retrying in {delay:.1f}s …"
                )
                await asyncio.sleep(delay)

        # Should not reach here, but raise last exception if we do
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    async def _generate_single_impl(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Internal implementation of single request generation (no retry)."""
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
        """Generate completions for a single request with retry logic."""
        return await self._retry_with_backoff_async(
            lambda: self._generate_single_impl(request, params),
            context=f"generate model={self.model_name}",
        )

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
