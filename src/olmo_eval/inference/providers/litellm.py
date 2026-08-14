"""LiteLLM provider for API-based inference."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider
from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.retry import retry_with_backoff
from olmo_eval.inference.tokenizer_utils import resolve_prompt_truncation_limit, resolve_truncation_side
from olmo_eval.inference.utils import run_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Maximum stop sequences supported by OpenAI-compatible APIs
_MAX_STOP_SEQUENCES = 4

logger = get_logger(__name__)


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
        "LITELLM_PROXY_API_KEY": "litellm_proxy_api_key",
    }

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_base: str | None = None,
        max_concurrency: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **api_kwargs,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier (e.g., "gpt-4", "claude-3-opus").
            base_url: Optional base URL for OpenAI-compatible endpoints.
            api_base: Optional API base for litellm.completion.
            max_concurrency: Maximum number of concurrent API requests.
            max_retries: Maximum number of retries for transient errors.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            **api_kwargs: Additional arguments passed to litellm.completion.
        """
        try:
            import litellm  # type: ignore[ty:unresolved-import]
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMProvider. Install with: uv pip install litellm"
            ) from e

        super().__init__(model_name)
        self._litellm = litellm
        self.base_url = base_url
        self.api_base = api_base
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_kwargs = api_kwargs
        self._client: AsyncOpenAI | None = None  # Cached client

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

    def get_openai_client(self) -> AsyncOpenAI | None:
        """Get an AsyncOpenAI client if base_url is configured.

        Returns cached client on subsequent calls to avoid connection pool leaks.

        Returns:
            AsyncOpenAI client if base_url is set, None otherwise.
        """
        if self.base_url is None:
            return None

        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            )

        return self._client

    def _max_input_tokens(self, request: LMRequest, params: SamplingParams) -> int | None:
        """Best-effort input budget for ``truncate_prompt_tokens=-1``."""
        if request.max_length is not None:
            reserved_output = params.max_tokens if params.max_tokens is not None else 0
            return max(request.max_length - reserved_output, 1)

        try:
            model_info = self._litellm.get_model_info(self.model_name)
        except Exception:
            return None
        max_input_tokens = model_info.get("max_input_tokens") if isinstance(model_info, dict) else None
        return max_input_tokens if isinstance(max_input_tokens, int) and max_input_tokens > 0 else None

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
        request: LMRequest,
        params: SamplingParams,
    ) -> list[dict[str, Any]]:
        """Truncate message content while preserving a valid chat-message structure."""
        if params.truncate_prompt_tokens is None:
            return [dict(message) for message in messages]

        limit = resolve_prompt_truncation_limit(
            params.truncate_prompt_tokens,
            max_input_tokens=self._max_input_tokens(request, params),
        )
        assert limit is not None
        side = resolve_truncation_side(params.truncation_side, default="right")
        truncated = [dict(message) for message in messages]

        def token_count() -> int:
            return int(self._litellm.token_counter(model=self.model_name, messages=truncated))

        current_count = token_count()
        if current_count <= limit:
            return truncated

        indices = range(len(truncated)) if side == "left" else range(len(truncated) - 1, -1, -1)
        for index in indices:
            content = truncated[index].get("content")
            if not isinstance(content, str) or not content:
                continue
            token_ids = list(self._litellm.encode(model=self.model_name, text=content))
            if not token_ids:
                continue
            excess = max(current_count - limit, 0)
            remove_count = min(excess, len(token_ids))
            if remove_count <= 0:
                break
            kept = token_ids[remove_count:] if side == "left" else token_ids[:-remove_count]
            truncated[index]["content"] = self._litellm.decode(model=self.model_name, tokens=kept)
            current_count = token_count()
            if current_count <= limit:
                return truncated

        # Chat-template overhead can itself exceed the requested limit after all
        # textual content is exhausted. Drop complete messages from the requested
        # truncation side while retaining at least one structurally valid message.
        while current_count > limit and len(truncated) > 1:
            if side == "left":
                truncated.pop(0)
            else:
                truncated.pop()
            current_count = token_count()

        if current_count > limit:
            raise ValueError(
                f"LiteLLM prompt cannot fit within truncate_prompt_tokens={limit}; "
                f"chat-template overhead is {current_count} tokens."
            )
        return truncated

    async def _generate_single_impl(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        if request.messages:
            messages = [dict(m) for m in request.messages]
        else:
            messages = [{"role": "user", "content": request.prompt}]
        messages = self._truncate_messages(messages, request, params)

        kwargs: dict[str, Any] = {
            "api_base": self.api_base,
            "model": self.model_name,
            "messages": messages,
            "n": params.num_samples,
            **self.api_kwargs,
        }
        if params.max_tokens is not None:
            kwargs["max_completion_tokens"] = params.max_tokens

        kwargs["temperature"] = params.temperature
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)[:_MAX_STOP_SEQUENCES]
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1

        response = await self._litellm.acompletion(**kwargs)

        outputs = []
        for choice in response.choices:
            text = choice.message.content or ""

            logprob_entries: list[LogProbEntry] | None = None
            metadata: dict[str, Any] = {}
            logprobs_data = getattr(choice, "logprobs", None)
            if logprobs_data and hasattr(logprobs_data, "content") and logprobs_data.content:
                logprob_entries = []
                for lp in logprobs_data.content:
                    entry: LogProbEntry = {"token": lp.token, "logprob": lp.logprob}
                    lp_bytes = getattr(lp, "bytes", None)
                    if lp_bytes is not None:
                        entry["bytes"] = lp_bytes
                    logprob_entries.append(entry)

                sum_logits = sum(entry["logprob"] for entry in logprob_entries)
                num_tokens = len(logprob_entries)
                metadata = {
                    "sum_logits": sum_logits,
                    "num_tokens": num_tokens,
                    "num_tokens_all": num_tokens,
                }

            outputs.append(LMOutput(text=text, logprobs=logprob_entries, metadata=metadata))

        return outputs

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any] | None:
        params = self._default_sampling_params(sampling_params)
        trace = super().describe_request(request, sampling_params)
        if trace is None:
            return None

        if request.request_type == RequestType.LOGLIKELIHOOD:
            trace["provider"] = "LiteLLMProvider"
            trace["endpoint"] = "litellm.acompletion"
            trace["generation_kwargs"] = {
                "max_gen_toks": 50,
                "do_sample": False,
                "temperature": params.temperature,
                "logprobs": True,
                "top_logprobs": 1,
            }
            if params.truncate_prompt_tokens is not None:
                trace["generation_kwargs"]["truncate_prompt_tokens"] = params.truncate_prompt_tokens
            if params.truncation_side is not None:
                trace["generation_kwargs"]["truncation_side"] = params.truncation_side
            trace["stop_sequences"] = []
            return trace

        trace["provider"] = "LiteLLMProvider"
        trace["endpoint"] = "litellm.acompletion"
        trace["generation_kwargs"] = {
            "max_gen_toks": params.max_tokens,
            "do_sample": params.do_sample and params.temperature > 0,
            "temperature": params.temperature,
            "logprobs": True,
            "top_logprobs": 1,
            "num_samples": params.num_samples,
        }
        if params.top_p is not None:
            trace["generation_kwargs"]["top_p"] = params.top_p
        if params.truncate_prompt_tokens is not None:
            trace["generation_kwargs"]["truncate_prompt_tokens"] = params.truncate_prompt_tokens
        if params.truncation_side is not None:
            trace["generation_kwargs"]["truncation_side"] = params.truncation_side
        trace["stop_sequences"] = list(params.stop_sequences or ())[:_MAX_STOP_SEQUENCES]
        return trace

    async def _generate_single_async(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        return await retry_with_backoff(
            lambda: self._generate_single_impl(request, params),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            context=f"generate model={self.model_name}",
            sdk_module=self._litellm,
        )

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions."""
        from olmo_eval.common.progress import ProgressLogger
        from olmo_eval.inference.dispatch import dispatch_concurrent

        logger.info(
            f"Sending {len(requests)} requests for {self.model_name}"
            f" with max_concurrency {self.max_concurrency}"
        )

        params = self._default_sampling_params(sampling_params)
        progress = ProgressLogger(total=len(requests), desc="Generating", logger=logger)

        async def process(req: LMRequest) -> list[LMOutput]:
            return await self._generate_single_async(req, params)

        def on_progress(done: int, total: int) -> None:
            progress.update(1)

        results = await dispatch_concurrent(
            requests,
            process,
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
            on_progress=on_progress,
        )
        progress.close()
        return [r if r is not None else [] for r in results]

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions (sync wrapper)."""
        return run_async(self.agenerate(requests, sampling_params))

    async def _logprobs_single_impl(
        self,
        request: LMRequest,
        params: SamplingParams | None = None,
    ) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        params = self._default_sampling_params(params)
        if request.messages:
            default_content = request.messages[0].get("content", "") if request.messages else ""
        else:
            default_content = request.prompt

        outputs = []
        cont_prompts = request.continuation_prompts
        for i, continuation in enumerate(request.continuations or ()):
            content = cont_prompts[i] if cont_prompts else default_content
            messages = self._truncate_messages(
                [{"role": "user", "content": content}], request, params
            )

            response = await self._litellm.acompletion(
                api_base=self.api_base,
                model=self.model_name,
                messages=messages,
                max_completion_tokens=50,
                temperature=params.temperature,
                logprobs=True,
                top_logprobs=1,
                **self.api_kwargs,
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

    async def _logprobs_single_async(
        self,
        request: LMRequest,
        params: SamplingParams | None = None,
    ) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        return await retry_with_backoff(
            lambda: self._logprobs_single_impl(request, params),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            context=f"logprobs model={self.model_name}",
            sdk_module=self._litellm,
        )

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations."""
        from olmo_eval.common.progress import ProgressLogger
        from olmo_eval.inference.dispatch import dispatch_concurrent

        logger.info(
            f"Sending {len(requests)} logprob requests for {self.model_name}"
            f" with max_concurrency {self.max_concurrency}"
        )

        params = self._default_sampling_params(sampling_params)
        progress = ProgressLogger(total=len(requests), desc="Logprobs", logger=logger)

        def on_progress(done: int, total: int) -> None:
            progress.update(1)

        results = await dispatch_concurrent(
            requests,
            lambda request: self._logprobs_single_async(request, params),
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
            on_progress=on_progress,
        )
        progress.close()
        return [r if r is not None else [] for r in results]

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Compute logprobs for continuations (sync wrapper)."""
        return run_async(self.alogprobs(requests, sampling_params))