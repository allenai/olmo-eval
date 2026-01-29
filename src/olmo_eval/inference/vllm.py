"""vLLM provider."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from olmo_eval.core import LMOutput, LMRequest, SamplingParams
from olmo_eval.core.types import LogProbEntry

from .base import InferenceProvider

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.outputs import RequestOutput


def _configure_vllm_logger(worker_id: str | None) -> None:
    """Configure vLLM's logger to include worker_id in output.

    Args:
        worker_id: Worker identifier to include in log format, or None to use default format.
    """
    vllm_logger = logging.getLogger("vllm")

    # Remove existing handlers to avoid duplicates
    for handler in vllm_logger.handlers[:]:
        vllm_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    if worker_id:
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s [{worker_id}] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    vllm_logger.addHandler(handler)
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.propagate = False


def _get_token_string(logprob_obj: Any, token_id: int, tokenizer: Any = None) -> str:
    """Extract token string from vLLM logprob object."""
    if hasattr(logprob_obj, "decoded_token"):
        return logprob_obj.decoded_token
    if tokenizer is not None:
        return tokenizer.decode([token_id])
    return str(token_id)


def _coerce_logprob_to_num(logprob: Any) -> float:
    """Handle both old (float) and new (Logprob object) vLLM versions."""
    return getattr(logprob, "logprob", logprob)


def _convert_logprobs(
    vllm_logprobs: list[dict[int, Any]] | None,
    tokenizer: Any = None,
) -> list[LogProbEntry] | None:
    """Convert vLLM logprobs format to standard format.

    Works with both old (float) and new (Logprob object) vLLM versions.
    """
    if vllm_logprobs is None:
        return None

    result: list[LogProbEntry] = []
    for token_logprobs in vllm_logprobs:
        if not token_logprobs:
            continue
        # vLLM returns dict of {token_id: LogprobInfo}, take first (chosen) token
        token_id, logprob_obj = next(iter(token_logprobs.items()))
        token_str = _get_token_string(logprob_obj, token_id, tokenizer)
        logprob_val = _coerce_logprob_to_num(logprob_obj)
        result.append(
            {
                "token": token_str,
                "logprob": logprob_val,
                "bytes": list(token_str.encode("utf-8")),
            }
        )

    return result


class VLLMProvider(InferenceProvider):
    """Provider using vLLM for high-throughput inference."""

    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        attention_backend: str | None = None,
        worker_id: str | None = None,
        **engine_kwargs,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: HuggingFace model identifier or local path.
            tokenizer: Tokenizer path/identifier. If not specified, uses the model path.
            attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN").
                If not specified, vLLM will auto-select based on available backends.
            worker_id: Optional worker identifier for logging. If provided, vLLM logs
                will include this identifier.
            **engine_kwargs: Additional arguments passed to vLLM LLM engine.
        """
        # Suppress verbose vLLM logging by default, but allow override
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        # Configure vLLM logger with worker_id if provided
        if worker_id:
            _configure_vllm_logger(worker_id)

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("vllm is required for VLLMProvider") from e

        super().__init__(model_name)
        self._worker_id = worker_id
        engine_kwargs.setdefault("gpu_memory_utilization", 0.8)

        # Configure attention backend if specified (e.g., FLASHINFER, FLASH_ATTN)
        if attention_backend:
            engine_kwargs.setdefault("attention_backend", attention_backend)

        # Use separate tokenizer if specified
        if tokenizer:
            engine_kwargs.setdefault("tokenizer", tokenizer)

        self.llm: LLM = LLM(model=model_name, **engine_kwargs)

    @property
    def max_length(self) -> int:
        """Get the maximum model context length."""
        if not hasattr(self, "_max_length"):
            self._max_length = self.llm.llm_engine.model_config.max_model_len
        return self._max_length

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Encode context and continuation separately (robust to non-additive tokenization).

        Matches lm_eval behavior: trailing spaces from context are moved to continuation
        before tokenization to ensure consistent token boundaries.
        """
        tokenizer = self.llm.get_tokenizer()

        # Match lm_eval behavior: move trailing spaces from context to continuation
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = tokenizer.encode(context + continuation, add_special_tokens=False)
        context_enc = tokenizer.encode(context, add_special_tokens=False)
        continuation_enc = whole_enc[len(context_enc) :]

        return context_enc, continuation_enc

    def _build_sampling_params(self, params: SamplingParams) -> Any:
        """Convert SamplingParams to vLLM SamplingParams."""
        from vllm import SamplingParams as VLLMSamplingParams

        kwargs: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "n": params.num_samples,
        }

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            kwargs["top_k"] = params.top_k
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        if params.logprobs is not None:
            kwargs["logprobs"] = params.logprobs

        return VLLMSamplingParams(**kwargs)

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        vllm_params = self._build_sampling_params(params)

        prompts = [req.prompt for req in requests]
        # Disable tqdm progress bar - we use our own worker-scoped logging
        outputs: list[RequestOutput] = self.llm.generate(prompts, vllm_params, use_tqdm=False)

        return [
            [
                LMOutput(
                    text=completion.text,
                    logprobs=_convert_logprobs(completion.logprobs),
                )
                for completion in output.outputs
            ]
            for output in outputs
        ]

    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        from vllm import SamplingParams as VLLMSamplingParams

        vllm_params = VLLMSamplingParams(
            prompt_logprobs=5,
            max_tokens=1,
            temperature=0.0,
        )

        tokenizer = self.llm.get_tokenizer()
        max_len = self.max_length

        # Build token sequences for all continuations
        token_inputs: list[list[int]] = []
        request_meta: list[tuple[int, int, int]] = []  # (ctxlen, num_tokens_all, overflow)

        for request in requests:
            continuations = request.continuations or ()
            for continuation in continuations:
                # Handle empty context: use BOS token as context
                if request.prompt == "":
                    bos_id = tokenizer.bos_token_id
                    if bos_id is None:
                        bos_id = tokenizer.eos_token_id
                    context_enc = [bos_id] if bos_id is not None else []
                    continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
                else:
                    context_enc, continuation_enc = self._encode_pair(request.prompt, continuation)

                # Calculate overflow and left-truncate to max_length - 1
                full_len = len(context_enc) + len(continuation_enc)
                overflow = full_len - (max_len - 1)
                inp = (context_enc + continuation_enc)[-(max_len - 1) :]

                # Adjust ctxlen based on overflow
                ctxlen = len(context_enc) - max(0, overflow)
                ctxlen = max(0, ctxlen)  # Ensure non-negative

                token_inputs.append(inp)
                request_meta.append((ctxlen, len(inp), overflow))

        # Call vLLM with token IDs instead of strings
        # Pass as list of dicts with prompt_token_ids key
        # Disable tqdm progress bar - we use our own worker-scoped logging
        prompts = [{"prompt_token_ids": tokens} for tokens in token_inputs]
        outputs: list[RequestOutput] = self.llm.generate(prompts, vllm_params, use_tqdm=False)

        # Parse results back to per-request structure
        output_iter = iter(outputs)
        meta_iter = iter(request_meta)
        tokens_iter = iter(token_inputs)
        results = []

        for request in requests:
            continuations = request.continuations or ()
            request_outputs = []

            for continuation in continuations:
                output = next(output_iter)
                ctxlen, num_tokens_all, overflow = next(meta_iter)
                inp = next(tokens_iter)

                logprob_entries = []
                total = 0.0
                is_greedy = True

                prompt_logprobs = output.prompt_logprobs or []
                # Skip the first ctxlen positions (context tokens)
                cont_logprobs = prompt_logprobs[ctxlen:] if ctxlen < len(prompt_logprobs) else []
                # Get continuation token IDs from the actual input
                cont_tokens = inp[ctxlen:]

                for token_id, token_probs in zip(cont_tokens, cont_logprobs, strict=True):
                    if not token_probs:
                        continue

                    # Look up logprob for the actual continuation token (not first key in dict)
                    lp_obj = token_probs.get(token_id)
                    if lp_obj is None:
                        continue
                    logprob_val = _coerce_logprob_to_num(lp_obj)

                    token_str = _get_token_string(lp_obj, token_id, tokenizer)
                    logprob_entries.append(
                        {
                            "token": token_str,
                            "logprob": logprob_val,
                            "bytes": list(token_str.encode("utf-8")),
                        }
                    )
                    total += logprob_val

                    # Check if this token is the argmax (greedy choice)
                    if is_greedy:
                        max_token_id = max(
                            token_probs.keys(),
                            key=lambda tid: _coerce_logprob_to_num(token_probs[tid]),
                        )
                        if max_token_id != token_id:
                            is_greedy = False

                num_tokens = len(logprob_entries)
                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=logprob_entries,
                        metadata={
                            "total_logprob": total,
                            "sum_logits": total,  # Alias for compatibility
                            "num_tokens": num_tokens,
                            "num_tokens_all": num_tokens_all,
                            "is_greedy": is_greedy,
                        },
                    )
                )

            results.append(request_outputs)

        return results


class AsyncVLLMProvider:
    """Async vLLM provider with continuous batching for streaming results.

    Uses vLLM's AsyncLLM (V1 engine) or AsyncLLMEngine (legacy) to enable
    true continuous batching where requests can be added while others are
    processing, and results stream back as they complete.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        attention_backend: str | None = None,
        worker_id: str | None = None,
        **engine_kwargs,
    ) -> None:
        """Initialize the async provider.

        Args:
            model_name: HuggingFace model identifier or local path.
            tokenizer: Tokenizer path/identifier. If not specified, uses the model path.
            attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN").
                If not specified, vLLM will auto-select based on available backends.
            worker_id: Optional worker identifier for logging. If provided, vLLM logs
                will include this identifier.
            **engine_kwargs: Additional arguments passed to vLLM engine.
        """
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        # Configure vLLM logger with worker_id if provided
        if worker_id:
            _configure_vllm_logger(worker_id)

        self._worker_id = worker_id

        self.model_name = model_name
        engine_kwargs.setdefault("gpu_memory_utilization", 0.8)

        # Configure attention backend if specified
        if attention_backend:
            engine_kwargs.setdefault("attention_backend", attention_backend)

        # Use separate tokenizer if specified
        if tokenizer:
            engine_kwargs.setdefault("tokenizer", tokenizer)

        # Try V1 engine first (vLLM 0.6.0+), fall back to legacy AsyncLLMEngine
        self._use_v1_engine = False
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            engine_args = AsyncEngineArgs(model=model_name, **engine_kwargs)
            self.engine: Any = AsyncLLM.from_engine_args(engine_args)
            self._use_v1_engine = True
        except ImportError:
            # Fall back to legacy AsyncLLMEngine
            try:
                from vllm import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine

                engine_args = AsyncEngineArgs(model=model_name, **engine_kwargs)
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            except ImportError as e:
                raise ImportError("vllm is required for AsyncVLLMProvider") from e

        self._request_counter = 0
        # Store pending requests: request_id -> (prompt, vllm_params)
        self._pending_requests: dict[str, tuple[str, Any]] = {}

    def _get_next_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"req-{self._request_counter}"

    def _build_sampling_params(self, params: SamplingParams) -> Any:
        """Convert SamplingParams to vLLM SamplingParams."""
        from vllm import SamplingParams as VLLMSamplingParams

        return VLLMSamplingParams(
            max_tokens=params.max_tokens,
            n=params.num_samples,
            temperature=params.temperature if params.temperature > 0 else 0.0,
            top_p=params.top_p if params.top_p else 1.0,
            top_k=params.top_k if params.top_k else -1,
            stop=list(params.stop_sequences) if params.stop_sequences else None,
            logprobs=params.logprobs,
        )

    async def add_request(
        self,
        request_id: str,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """Add a single request to be processed (non-blocking).

        Args:
            request_id: Unique identifier for this request.
            request: The LM request to process.
            sampling_params: Optional sampling parameters.

        Raises:
            ValueError: If request type is LOGLIKELIHOOD (not supported in streaming mode).
        """
        from olmo_eval.core import RequestType

        # Streaming backend only supports COMPLETION requests
        # LOGLIKELIHOOD/BPB tasks require prompt_logprobs which isn't supported here
        if request.request_type == RequestType.LOGLIKELIHOOD:
            raise ValueError(
                "LOGLIKELIHOOD requests (e.g., :bpb tasks) are not supported in "
                "--async-stream mode. Use --async or default sequential mode instead."
            )

        if not request.prompt:
            raise ValueError(
                "Empty prompts are not supported in --async-stream mode. "
                "This may be a :bpb task - use --async or default sequential mode instead."
            )

        params = sampling_params or SamplingParams()
        vllm_params = self._build_sampling_params(params)
        # Store for later processing in stream_results
        self._pending_requests[request_id] = (request.prompt, vllm_params)

    async def stream_results(self) -> AsyncIterator[tuple[str, list[LMOutput]]]:
        """Stream results as they complete.

        Processes all pending requests concurrently and yields results
        as each request completes.

        Yields:
            Tuples of (request_id, list of LMOutput for that request).
        """
        import asyncio

        if not self._pending_requests:
            return

        async def process_single_request(
            request_id: str, prompt: str, params: Any
        ) -> tuple[str, list[LMOutput]] | None:
            """Process a single request and return when complete."""
            # Use AsyncLLMEngine.generate() with positional args: prompt, params, request_id
            async for output in self.engine.generate(prompt, params, request_id):
                if output.finished:
                    outputs = [
                        LMOutput(
                            text=completion.text,
                            logprobs=_convert_logprobs(completion.logprobs),
                        )
                        for completion in output.outputs
                    ]
                    return request_id, outputs
            return None

        # Create tasks for all pending requests
        tasks = [
            asyncio.create_task(process_single_request(req_id, prompt, params))
            for req_id, (prompt, params) in list(self._pending_requests.items())
        ]

        # Yield results as they complete
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                request_id, outputs = result
                self._pending_requests.pop(request_id, None)
                yield request_id, outputs

    async def shutdown(self) -> None:
        """Shutdown the engine gracefully."""
        import asyncio

        if self.engine is None:
            return

        # Try different shutdown methods depending on vLLM version
        if hasattr(self.engine, "shutdown"):
            try:
                result = self.engine.shutdown()
                # Handle both sync and async shutdown methods
                if result is not None and (asyncio.iscoroutine(result) or asyncio.isfuture(result)):
                    await result
            except Exception:
                pass  # Ignore shutdown errors
