"""vLLM provider."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider, is_debug_requests
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

logger = logging.getLogger(__name__)

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
        # Set vLLM logging level - DEBUG if OLMO_EVAL_DEBUG_PROVIDER=1, otherwise WARNING
        if is_debug_provider():
            os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        else:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        # Configure vLLM logger with worker_id if provided
        if worker_id:
            _configure_vllm_logger(worker_id)

        try:
            from vllm import LLM
        except ImportError as e:
            import traceback

            logger.error(f"Failed to import vllm: {e}")
            logger.error(traceback.format_exc())
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

        # Disable tqdm loading bar by default, enable with --debug-provider
        engine_kwargs.setdefault("use_tqdm_on_load", is_debug_provider())

        # Extract add_bos_token before passing to LLM (not a valid vLLM EngineArgs parameter).
        # When False, prompts will be pre-tokenized without special tokens and passed as token IDs,
        # matching the old framework's behavior (tokenizer(text, add_special_tokens=False)).
        self._add_bos_token: bool | None = engine_kwargs.pop("add_bos_token", None)

        self.llm: LLM = LLM(model=model_name, **engine_kwargs)

    @property
    def max_length(self) -> int:
        """Get the maximum model context length."""
        if not hasattr(self, "_max_length"):
            self._max_length = self.llm.llm_engine.model_config.max_model_len
        return self._max_length

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider."""
        return self.llm.get_tokenizer()

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

        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            kwargs["top_k"] = params.top_k
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        # Always request logprobs (default to 1) for metrics computation
        kwargs["logprobs"] = params.logprobs if params.logprobs is not None else 1

        return VLLMSamplingParams(**kwargs)

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        vllm_params = self._build_sampling_params(params)

        prompt_strs = [req.prompt for req in requests]

        if is_debug_requests():
            for i, prompt in enumerate(prompt_strs):
                logger.info(f"Prompt {i}:\n{prompt}")

        # When add_bos_token=False, pre-tokenize without special tokens and pass token IDs.
        # This bypasses vLLM's internal tokenization, matching the old framework behavior of
        # calling tokenizer(text, add_special_tokens=False) before passing to vLLM.
        if self._add_bos_token is False:
            tokenizer = self.llm.get_tokenizer()
            vllm_prompts: list = [
                {"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)}
                for p in prompt_strs
            ]
        else:
            vllm_prompts = prompt_strs

        # Disable tqdm progress bar - we use our own worker-scoped logging
        outputs: list[RequestOutput] = self.llm.generate(vllm_prompts, vllm_params, use_tqdm=False)

        results: list[list[LMOutput]] = []
        for output in outputs:
            request_outputs: list[LMOutput] = []
            for completion in output.outputs:
                logprobs = _convert_logprobs(completion.logprobs)

                # Compute metadata from logprobs
                metadata: dict[str, Any] = {}
                if logprobs:
                    sum_logits = sum(entry.get("logprob", 0.0) for entry in logprobs)
                    num_tokens = len(logprobs)
                    metadata = {
                        "sum_logits": sum_logits,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }

                request_outputs.append(
                    LMOutput(
                        text=completion.text,
                        logprobs=logprobs,
                        metadata=metadata,
                    )
                )
            results.append(request_outputs)

        return results

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
                context_enc, continuation_enc = encode_context_and_continuation(
                    tokenizer, request.prompt, continuation
                )

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

        if is_debug_requests():
            logger.info(f"vLLM logprobs: {len(prompts)} continuations")
            logger.info(f"Sampling params: {vllm_params}")

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

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions.

        Runs the synchronous vLLM generate in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Runs the synchronous vLLM logprobs in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return await asyncio.to_thread(self.logprobs, requests)
