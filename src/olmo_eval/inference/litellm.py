"""LiteLLM provider for API-based inference."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from olmo_eval.core.debug import is_debug_provider
from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import LMOutput, LMRequest, LogProbEntry, SamplingParams

from .base import InferenceProvider

# Maximum stop sequences supported by OpenAI-compatible APIs
_MAX_STOP_SEQUENCES = 4

# HTTP status codes that should trigger a retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# litellm exception type names that should never be retried
_NEVER_RETRY_TYPES = (
    "AuthenticationError",  # 401 – bad API key
    "BadRequestError",  # 400 – invalid params, content policy, etc.
    "NotFoundError",  # 404 – wrong model / endpoint
    "UnprocessableEntityError",  # 422 – semantic validation failure
)

# litellm exception type names that are always transient and should be retried
_ALWAYS_RETRY_TYPES = (
    "RateLimitError",  # 429
    "Timeout",  # request timed out
    "APIConnectionError",  # connection-level failure
    "ServiceUnavailableError",  # 503
    "InternalServerError",  # 500
)

logger = get_logger(__name__)

T = TypeVar("T")


class LiteLLMProvider(InferenceProvider):
    """Provider using LiteLLM for unified API access to various providers."""

    # Environment variable to LiteLLM attribute mappings
    _API_KEY_MAPPINGS = {
        "OPENAI_API_KEY": "openai_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "COHERE_API_KEY": "cohere_api_key",
        "TOGETHER_API_KEY": "together_api_key",
        "AZURE_API_KEY": "azure_api_key",
        "AZURE_API_BASE": "azure_api_base",
        "AZURE_API_VERSION": "azure_api_version",
    }

    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **api_kwargs,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier (e.g., "gpt-4", "claude-3-opus").
            max_concurrency: Maximum number of concurrent API requests.
            max_retries: Maximum number of retries for transient errors.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            **api_kwargs: Additional arguments passed to litellm.completion.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMProvider. Install with: uv pip install litellm"
            ) from e

        super().__init__(model_name)
        self._litellm = litellm
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_kwargs = api_kwargs

        if is_debug_provider():
            litellm._turn_on_debug()
        else:
            litellm.suppress_debug_info = True

        self._setup_api_keys()

    def _setup_api_keys(self) -> None:
        """Configure LiteLLM with API keys from environment."""
        for env_var, litellm_attr in self._API_KEY_MAPPINGS.items():
            value = os.getenv(env_var)
            if value:
                setattr(self._litellm, litellm_attr, value)

    def _is_retryable(self, exc: Exception) -> bool:
        """Determine whether *exc* should be retried.

        Uses litellm's typed exception hierarchy to classify errors:
        - Never retry: AuthenticationError, BadRequestError, NotFoundError,
          UnprocessableEntityError
        - Always retry: RateLimitError, Timeout, APIConnectionError,
          ServiceUnavailableError, InternalServerError
        - Falls back to HTTP status code for unknown subtypes.
        """
        # Never retry these – the request itself is wrong
        for attr in _NEVER_RETRY_TYPES:
            cls = getattr(self._litellm, attr, None)
            if cls is not None and isinstance(exc, cls):
                return False

        # Always retry these – transient server/network issues
        for attr in _ALWAYS_RETRY_TYPES:
            cls = getattr(self._litellm, attr, None)
            if cls is not None and isinstance(exc, cls):
                return True

        # Fall back to HTTP status code for any litellm error subtypes
        # not explicitly listed above; unknown / non-litellm exceptions are not retried
        status = getattr(exc, "status_code", None)
        return status is not None and int(status) in _RETRYABLE_STATUS_CODES

    @staticmethod
    def _format_error(exc: Exception) -> str:
        """Build a detailed, single-log-entry description of *exc*."""
        parts: list[str] = [f"  type: {type(exc).__qualname__}"]

        status = getattr(exc, "status_code", None)
        if status is not None:
            parts.append(f"  status_code: {status}")
        for attr in ("llm_provider", "model"):
            val = getattr(exc, attr, None)
            if val is not None:
                parts.append(f"  {attr}: {val}")

        message = getattr(exc, "message", None) or str(exc)
        # Truncate very long messages (e.g. full HTML error pages)
        if len(message) > 500:
            message = message[:500] + "…"
        parts.append(f"  message: {message}")

        # The wrapped cause often has the real reason (e.g. httpx.ConnectError)
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
            context: Optional human-readable label (e.g. ``"generate model=gpt-4"``)
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
                auth_cls = getattr(self._litellm, "AuthenticationError", None)
                if auth_cls is not None and isinstance(e, auth_cls):
                    logger.error(
                        f"Authentication failed{ctx}:\n{detail}\n"
                        f"  Verify the API key environment variable is set correctly."
                    )
                    raise

                # Not-found errors: fail immediately with actionable guidance
                not_found_cls = getattr(self._litellm, "NotFoundError", None)
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

    async def _generate_single_async(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        # Build messages from request
        if request.messages:
            messages = [dict(m) for m in request.messages]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # Prepare API kwargs
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "n": params.num_samples,
            "max_completion_tokens": params.max_tokens,
            **self.api_kwargs,
        }

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)[:_MAX_STOP_SEQUENCES]
        if params.logprobs is not None:
            kwargs["logprobs"] = True

        response = await self._litellm.acompletion(**kwargs)

        outputs = []
        for choice in response.choices:
            text = choice.message.content or ""

            # Convert logprobs to standard format
            logprob_entries: list[LogProbEntry] | None = None
            logprobs_data = getattr(choice, "logprobs", None)
            if logprobs_data and hasattr(logprobs_data, "content") and logprobs_data.content:
                logprob_entries = []
                for lp in logprobs_data.content:
                    entry: LogProbEntry = {"token": lp.token, "logprob": lp.logprob}
                    lp_bytes = getattr(lp, "bytes", None)
                    if lp_bytes is not None:
                        entry["bytes"] = lp_bytes
                    logprob_entries.append(entry)

            outputs.append(LMOutput(text=text, logprobs=logprob_entries))

        return outputs

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        from tqdm import tqdm

        logger.info(
            f"Sending {len(requests)} requests for {self.model_name}"
            f" with max_concurrency {self.max_concurrency}"
        )

        params = self._default_sampling_params(sampling_params)

        async def arun() -> list[list[LMOutput]]:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            pbar = tqdm(total=len(requests), desc="Processing instances", unit="inst")

            async def process(req: LMRequest) -> list[LMOutput]:
                async with semaphore:
                    result = await self._retry_with_backoff_async(
                        lambda r=req: self._generate_single_async(r, params),
                        context=f"generate model={self.model_name}",
                    )
                    pbar.update(1)
                    return result

            results = await asyncio.gather(*[process(r) for r in requests])
            pbar.close()
            return list(results)

        return asyncio.run(arun())

    async def _logprobs_single_async(self, request: LMRequest) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        if request.messages:
            content = request.messages[0].get("content", "") if request.messages else ""
        else:
            content = request.prompt

        response = await self._litellm.acompletion(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=50,
            temperature=0.0,
            logprobs=True,
            **self.api_kwargs,
        )

        # Extract logprobs from response
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

        # Map to continuations
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

    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        """Compute logprobs for continuations.

        Note: Most API providers don't support true continuation logprobs.
        This implementation provides an approximation by generating a response
        and returning those logprobs.
        """
        from tqdm import tqdm

        async def arun() -> list[list[LMOutput]]:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            pbar = tqdm(total=len(requests), desc="Processing instances", unit="inst")

            async def process(req: LMRequest) -> list[LMOutput]:
                async with semaphore:
                    result = await self._retry_with_backoff_async(
                        lambda r=req: self._logprobs_single_async(r),
                        context=f"logprobs model={self.model_name}",
                    )
                    pbar.update(1)
                    return result

            results = await asyncio.gather(*[process(r) for r in requests])
            pbar.close()
            return list(results)

        return asyncio.run(arun())
