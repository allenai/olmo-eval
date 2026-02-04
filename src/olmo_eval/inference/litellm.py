"""LiteLLM provider for API-based inference."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import LMOutput, LMRequest, LogProbEntry, SamplingParams

from .base import InferenceProvider

# Maximum stop sequences supported by OpenAI-compatible APIs
_MAX_STOP_SEQUENCES = 4

# HTTP status codes that should trigger a retry
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

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

        # Suppress verbose litellm debug info messages
        litellm.suppress_debug_info = True

        super().__init__(model_name)
        self._litellm = litellm
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_kwargs = api_kwargs
        self._setup_api_keys()

    def _setup_api_keys(self) -> None:
        """Configure LiteLLM with API keys from environment."""
        for env_var, litellm_attr in self._API_KEY_MAPPINGS.items():
            value = os.getenv(env_var)
            if value:
                setattr(self._litellm, litellm_attr, value)

    def _retry_with_backoff(self, func: Callable[[], T]) -> T:
        """Execute with exponential backoff for retryable errors.

        Retries on rate limits (429) and transient server errors (502, 503, 504, timeout).

        Args:
            func: Callable to execute.

        Returns:
            Result of the function call.

        Raises:
            Exception: If all retries are exhausted or non-retryable error occurs.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e

                # Check if this is a retryable error
                is_retryable = False
                error_str = str(e).lower()

                # Check for rate limit or timeout in error message
                if "rate" in error_str or "timeout" in error_str or "timed out" in error_str:
                    is_retryable = True

                # Check for HTTP status codes in error
                for status_code in _RETRYABLE_STATUS_CODES:
                    if str(status_code) in str(e):
                        is_retryable = True
                        break

                # Check for litellm-specific exceptions
                if hasattr(self._litellm, "RateLimitError") and isinstance(
                    e, self._litellm.RateLimitError
                ):
                    is_retryable = True
                if hasattr(self._litellm, "Timeout") and isinstance(e, self._litellm.Timeout):
                    is_retryable = True

                if not is_retryable or attempt >= self.max_retries:
                    raise

                # Exponential backoff with jitter
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Retryable error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # Should not reach here, but raise last exception if we do
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    def _generate_single(self, request: LMRequest, params: SamplingParams) -> list[LMOutput]:
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

        response = self._litellm.completion(**kwargs)

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

        params = self._default_sampling_params(sampling_params)

        def generate_with_retry(req: LMRequest) -> list[LMOutput]:
            return self._retry_with_backoff(lambda: self._generate_single(req, params))

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            results = list(
                tqdm(
                    executor.map(generate_with_retry, requests),
                    total=len(requests),
                    desc="Processing instances",
                    unit="inst",
                )
            )

        return results

    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        """Compute logprobs for continuations.

        Note: Most API providers don't support true continuation logprobs.
        This implementation provides an approximation by generating a response
        and returning those logprobs.
        """

        def _logprobs_single(request: LMRequest) -> list[LMOutput]:
            if request.messages:
                content = request.messages[0].get("content", "") if request.messages else ""
            else:
                content = request.prompt

            response = self._litellm.completion(
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
                    sum(lp["logprob"] for lp in completion_logprobs[:5])
                    if completion_logprobs
                    else 0.0
                )
                outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=completion_logprobs[:5] if completion_logprobs else None,
                        metadata={"total_logprob": total},
                    )
                )

            return outputs

        def logprobs_with_retry(req: LMRequest) -> list[LMOutput]:
            return self._retry_with_backoff(lambda: _logprobs_single(req))

        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            results = list(
                tqdm(
                    executor.map(logprobs_with_retry, requests),
                    total=len(requests),
                    desc="Processing instances",
                    unit="inst",
                )
            )

        return results
