"""vLLM provider."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider, is_debug_requests
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.hf_cache import refresh_hf_cache
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation, truncate_token_ids

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
        force_download: bool = False,
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
            force_download: Force-refresh Hugging Face model/tokenizer cache entries
                before initializing vLLM.
            **engine_kwargs: Additional arguments passed to vLLM LLM engine.
        """
        if is_debug_provider():
            os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        else:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

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
        if force_download:
            model_revision = engine_kwargs.get("revision")
            cache_dir = engine_kwargs.get("download_dir") or engine_kwargs.get("cache_dir")
            token = engine_kwargs.get("token")
            refresh_hf_cache(
                model_name,
                revision=model_revision,
                cache_dir=cache_dir,
                token=token,
                force_download=True,
            )
            if tokenizer and tokenizer != model_name:
                tokenizer_revision = engine_kwargs.get("tokenizer_revision") or model_revision
                refresh_hf_cache(
                    tokenizer,
                    revision=tokenizer_revision,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=True,
                )

        engine_kwargs.setdefault("gpu_memory_utilization", 0.8)
        if attention_backend:
            engine_kwargs.setdefault("attention_backend", attention_backend)
        if tokenizer:
            engine_kwargs.setdefault("tokenizer", tokenizer)
        engine_kwargs.setdefault("use_tqdm_on_load", is_debug_provider())

        # Extract add_bos_token before passing to LLM (not a valid vLLM EngineArgs parameter).
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

    def _max_input_tokens(self, params: SamplingParams) -> int:
        reserved_output = params.max_tokens if params.max_tokens is not None else 1
        return max(self.max_length - reserved_output, 1)

    def _prompt_token_ids(self, prompt: str, params: SamplingParams) -> list[int]:
        tokenizer = self.llm.get_tokenizer()
        encode_kwargs: dict[str, Any] = {}
        if self._add_bos_token is not None:
            encode_kwargs["add_special_tokens"] = self._add_bos_token
        token_ids = tokenizer.encode(prompt, **encode_kwargs)
        return truncate_token_ids(
            token_ids,
            params.truncate_prompt_tokens,
            params.truncation_side,
            tokenizer=tokenizer,
            max_input_tokens=self._max_input_tokens(params),
        )

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Encode context and continuation separately (robust to non-additive tokenization)."""
        tokenizer = self.llm.get_tokenizer()
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

        temperature = 0.0 if not params.do_sample else params.temperature
        top_p = None if not params.do_sample else params.top_p
        top_k = None if not params.do_sample else params.top_k
        kwargs: dict[str, Any] = {"max_tokens": params.max_tokens, "n": params.num_samples}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        kwargs["logprobs"] = params.logprobs if params.logprobs is not None else 1
        return VLLMSamplingParams(**kwargs)

    def _format_prompt(self, request: LMRequest) -> str:
        """Format an LMRequest into the text prompt expected by inline vLLM."""
        if request.request_type == RequestType.CHAT and request.messages:
            tokenizer = self.llm.get_tokenizer()
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("CHAT requests require a tokenizer with apply_chat_template")
            return tokenizer.apply_chat_template(
                list(request.messages), tokenize=False, add_generation_prompt=True
            )
        return request.prompt

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        vllm_params = self._build_sampling_params(params)
        prompt_strs = [self._format_prompt(req) for req in requests]

        if is_debug_requests():
            for i, prompt in enumerate(prompt_strs):
                logger.info(f"Prompt {i}:\n{prompt}")

        # Prompt truncation requires token IDs so the caller-requested side can be
        # applied before vLLM sees the request. add_bos_token=False already uses this path.
        if params.truncate_prompt_tokens is not None or self._add_bos_token is False:
            vllm_prompts: list = [
                {"prompt_token_ids": self._prompt_token_ids(prompt, params)} for prompt in prompt_strs
            ]
        else:
            vllm_prompts = prompt_strs

        outputs: list[RequestOutput] = self.llm.generate(vllm_prompts, vllm_params, use_tqdm=False)

        results: list[list[LMOutput]] = []
        for output in outputs:
            request_outputs: list[LMOutput] = []
            for completion in output.outputs:
                logprobs = _convert_logprobs(completion.logprobs)
                metadata: dict[str, Any] = {}
                if logprobs:
                    sum_logits = sum(entry.get("logprob", 0.0) for entry in logprobs)
                    num_tokens = len(logprobs)
                    metadata = {
                        "sum_logits": sum_logits,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }
                request_outputs.append(LMOutput(text=completion.text, logprobs=logprobs, metadata=metadata))
            results.append(request_outputs)
        return results

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
            trace["provider"] = "VLLMProvider"
            trace["endpoint"] = "llm.generate"
            trace["generation_kwargs"] = {
                "max_gen_toks": 1,
                "do_sample": False,
                "temperature": params.temperature,
                "prompt_logprobs": 1,
            }
            if params.truncate_prompt_tokens is not None:
                trace["generation_kwargs"]["truncate_prompt_tokens"] = params.truncate_prompt_tokens
            if params.truncation_side is not None:
                trace["generation_kwargs"]["truncation_side"] = params.truncation_side
            trace["stop_sequences"] = []
            trace["input_mode"] = "prompt_token_ids"
            return trace

        vllm_params = self._build_sampling_params(params)
        trace["provider"] = "VLLMProvider"
        trace["endpoint"] = "llm.generate"
        trace["generation_kwargs"] = {
            "max_gen_toks": vllm_params.max_tokens,
            "do_sample": params.do_sample and params.temperature > 0,
            "temperature": getattr(vllm_params, "temperature", params.temperature),
            "logprobs": getattr(vllm_params, "logprobs", None),
            "num_samples": vllm_params.n,
        }
        if getattr(vllm_params, "top_p", None) is not None:
            trace["generation_kwargs"]["top_p"] = vllm_params.top_p
        if getattr(vllm_params, "top_k", None) is not None:
            trace["generation_kwargs"]["top_k"] = vllm_params.top_k
        if params.truncate_prompt_tokens is not None:
            trace["generation_kwargs"]["truncate_prompt_tokens"] = params.truncate_prompt_tokens
        if params.truncation_side is not None:
            trace["generation_kwargs"]["truncation_side"] = params.truncation_side
        trace["stop_sequences"] = list(params.stop_sequences or ())
        trace["input_mode"] = (
            "prompt_token_ids"
            if params.truncate_prompt_tokens is not None or self._add_bos_token is False
            else "text"
        )
        return trace

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        from vllm import SamplingParams as VLLMSamplingParams

        params = self._default_sampling_params(sampling_params)
        vllm_params = VLLMSamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            temperature=params.temperature,
        )

        tokenizer = self.llm.get_tokenizer()
        default_max_len = self.max_length
        token_inputs: list[list[int]] = []
        request_meta: list[tuple[int, int, int]] = []

        for request in requests:
            max_len = request.max_length or default_max_len
            continuations = request.continuations or ()
            cont_prompts = request.continuation_prompts
            for i, continuation in enumerate(continuations):
                prompt = cont_prompts[i] if cont_prompts else request.prompt
                context_enc, continuation_enc = encode_context_and_continuation(
                    tokenizer, prompt, continuation
                )
                context_enc = truncate_token_ids(
                    context_enc,
                    params.truncate_prompt_tokens,
                    params.truncation_side,
                    tokenizer=tokenizer,
                    max_input_tokens=max(max_len - len(continuation_enc), 1),
                )

                full_len = len(context_enc) + len(continuation_enc)
                overflow = full_len - (max_len - 1)
                inp = (context_enc + continuation_enc)[-(max_len - 1) :]
                ctxlen = max(0, len(context_enc) - max(0, overflow))

                token_inputs.append(inp)
                request_meta.append((ctxlen, len(inp), overflow))

        prompts = [{"prompt_token_ids": tokens} for tokens in token_inputs]

        if is_debug_requests():
            logger.info(f"vLLM logprobs: {len(prompts)} continuations")
            logger.info(f"Sampling params: {vllm_params}")

        outputs: list[RequestOutput] = self.llm.generate(prompts, vllm_params, use_tqdm=False)
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
                logprob_entries: list[LogProbEntry] = []
                total = 0.0
                is_greedy = True

                prompt_logprobs = output.prompt_logprobs or []
                cont_logprobs = prompt_logprobs[ctxlen:] if ctxlen < len(prompt_logprobs) else []
                cont_tokens = inp[ctxlen:]

                for token_id, token_probs in zip(cont_tokens, cont_logprobs, strict=True):
                    if not token_probs:
                        continue
                    if is_greedy:
                        max_token_id = max(
                            token_probs.keys(), key=lambda tid: _coerce_logprob_to_num(token_probs[tid])
                        )
                        if max_token_id != token_id:
                            is_greedy = False
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

                num_tokens = len(logprob_entries)
                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=logprob_entries,
                        metadata={
                            "total_logprob": total,
                            "sum_logits": total,
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
        """Async generate completions."""
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations."""
        return await asyncio.to_thread(self.logprobs, requests, sampling_params)
