"""Inline SGLang provider for text and chat generation."""

from __future__ import annotations

import asyncio
import math
import os
import sys
from importlib import import_module
from typing import Any

from jinja2 import TemplateError

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import CONTEXT_OVERFLOW_KEY, REQUEST_ERROR_KEY
from olmo_eval.inference.hf_cache import refresh_hf_cache
from olmo_eval.inference.providers.sglang_profiles import prepare_profile

logger = get_logger(__name__)

# Keep this allowlist aligned with the pinned olmo-sglang arithmetic options.
# Record exact startup values (None means unset), rather than duplicating the
# plugin's defaults or claiming that an automatic/per-layer choice is resolved.
_ARITHMETIC_ENVIRONMENT = (
    "OLMO_SGLANG_CORE_COMPAT",
    "OLMO_SGLANG_ROUNDING_KERNELS",
    "OLMO_SGLANG_ROUTING_KERNELS",
    "OLMO_SGLANG_ROUTING_TAIL",
    "OLMO_SGLANG_ROUTER_INPUT",
    "OLMO_SGLANG_SHARED_ADD_NORM",
    "OLMO_SGLANG_KDA_PROJECTIONS",
    "OLMO_SGLANG_KDA_GATE_PROJECTIONS",
    "OLMO_SGLANG_MOE_PROJECTIONS",
    "OLMO_SGLANG_SAMPLER",
)


def _require_original_logprobs() -> None:
    """Establish raw model logprobs before plugins can import the sampler."""
    value = os.environ.setdefault("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    if value.lower() not in ("1", "true"):
        raise ValueError(
            "SGLangProvider requires raw model logprobs: set "
            "SGLANG_RETURN_ORIGINAL_LOGPROB=1 before starting the worker."
        )
    for name in (
        "sglang.srt.layers.sampler",
        "sglang.srt.speculative.standalone_worker_v2",
    ):
        module = sys.modules.get(name)
        if (
            module is not None
            and getattr(module, "SGLANG_RETURN_ORIGINAL_LOGPROB", None) is not True
        ):
            raise RuntimeError(
                f"{name} was already imported without raw model logprobs. "
                "Restart the worker with SGLANG_RETURN_ORIGINAL_LOGPROB=1; "
                "changing the environment after import cannot repair its cached convention."
            )


class SGLangProvider(InferenceProvider):
    """Run generation with a locally installed SGLang engine.

    Install the `sglang` extra in a compatible GPU runtime. Continuation likelihood
    scoring and tool requests are not supported by this provider.
    """

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        max_model_len: int | None = None,
        max_concurrency: int | None = None,
        tensor_parallel_size: int | None = None,
        add_bos_token: bool | None = None,
        model_registration: str | None = None,
        profile: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name)
        if kwargs.get("skip_tokenizer_init"):
            raise ValueError("SGLangProvider requires tokenizer initialization for chat and stops")
        if tensor_parallel_size is not None:
            if "tp_size" in kwargs and kwargs["tp_size"] != tensor_parallel_size:
                raise ValueError("tensor_parallel_size and tp_size must agree")
            kwargs["tp_size"] = tensor_parallel_size
        if force_download:
            refresh_hf_cache(
                model_name,
                revision=revision,
                cache_dir=kwargs.get("download_dir"),
                force_download=True,
            )
            if tokenizer and tokenizer != model_name:
                refresh_hf_cache(
                    tokenizer,
                    revision=revision,
                    cache_dir=kwargs.get("download_dir"),
                    force_download=True,
                )
        _require_original_logprobs()
        if model_registration:
            module, name = model_registration.split(":", 1)
            getattr(import_module(module), name)()
        options: dict[str, Any] = dict(
            kwargs, model_path=model_name, dtype=dtype, trust_remote_code=trust_remote_code
        )
        for key, value in (
            ("tokenizer_path", tokenizer),
            ("revision", revision),
            ("context_length", max_model_len),
            ("max_running_requests", max_concurrency),
        ):
            if value is not None:
                options[key] = value
        self.profile_report = prepare_profile(profile, options, model_registration)
        options = {**self.profile_report["defaults"], **options}
        self.profile_report["arithmetic_environment"] = {
            name: os.environ.get(name) for name in _ARITHMETIC_ENVIRONMENT
        }
        logger.info("SGLang profile: %s", self.profile_report)
        self._add_bos_token = add_bos_token
        sglang = import_module("sglang")
        _require_original_logprobs()
        self.engine = sglang.Engine(**options)
        self.recommended_batch_sequences: int | None = (
            options["max_running_requests"]
            if self.profile_report["selected"] != "generic"
            else None
        )
        self.profile_report["effective_options"] = {
            key: options[key]
            for key in (
                "max_running_requests",
                "max_total_tokens",
                "max_mamba_cache_size",
                "mem_fraction_static",
                "disable_radix_cache",
                "chunked_prefill_size",
                "cuda_graph_backend_decode",
                "cuda_graph_max_bs_decode",
                "cuda_graph_bs_decode",
                "cuda_graph_backend_prefill",
                "mamba_radix_cache_strategy",
                "context_length",
                "attention_backend",
                "sampling_backend",
                "moe_runner_backend",
                "moe_a2a_backend",
            )
            if key in options
        }
        # ServerArgs resolves defaults that were omitted from the constructor,
        # including the generic/fallback path. Avoid importing GPU modules here.
        server_args = getattr(self.engine.tokenizer_manager, "server_args", None)
        for key in (
            "attention_backend",
            "sampling_backend",
            "disable_radix_cache",
            "moe_runner_backend",
            "moe_a2a_backend",
        ):
            value = getattr(server_args, key, None)
            if type(value) in (str, bool, int, float):
                self.profile_report["effective_options"][key] = value
        if self.recommended_batch_sequences is not None:
            # Persist resolved capacity as well as requested settings; state pools
            # can lower admission. Older runtimes may not expose this diagnostic.
            try:
                info = self.engine.get_server_info()
                states = info.get("internal_states", []) if isinstance(info, dict) else []
                if len(states) == 1 and isinstance(states[0], dict):
                    memory = states[0].get("memory_usage")
                    if isinstance(memory, dict):
                        self.profile_report["engine_memory"] = memory
                    capacity = states[0].get("effective_max_running_requests_per_dp")
                    if type(capacity) is int and capacity > 0:
                        self.recommended_batch_sequences = min(
                            self.recommended_batch_sequences, capacity
                        )
                        self.profile_report["effective_max_running_requests"] = capacity
            except Exception:
                logger.debug("Resolved SGLang capacity unavailable", exc_info=True)
        self._closed = False
        self._capacity_checked = False
        self._previous_batch: tuple[int, int, int, int] | None = None

    def get_tokenizer(self) -> Any:
        """Return the engine's tokenizer."""
        return self.engine.tokenizer_manager.tokenizer

    def _validate_request(self, request: LMRequest) -> None:
        if request.images:
            raise NotImplementedError("SGLangProvider supports text-only requests")
        if request.tools or request.request_type == RequestType.LOGLIKELIHOOD:
            raise NotImplementedError("SGLangProvider supports text and chat generation only")
        if request.request_type == RequestType.CHAT and not request.messages:
            raise ValueError("CHAT requests require messages")

    def _format_prompt(self, request: LMRequest) -> str:
        self._validate_request(request)
        if request.request_type == RequestType.CHAT:
            return self.get_tokenizer().apply_chat_template(
                list(request.messages), tokenize=False, add_generation_prompt=True
            )
        return request.prompt

    def _prepare_request(self, request: LMRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = self._format_prompt(request)
        if self._add_bos_token is False:
            return {"input_ids": self.get_tokenizer().encode(prompt, add_special_tokens=False)}, {}
        return {"prompt": prompt}, {}

    def _build_sampling_params(self, params: SamplingParams) -> dict[str, Any]:
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            raise NotImplementedError("SGLangProvider does not support prompt truncation")
        if params.fit_max_tokens_to_context:
            raise NotImplementedError("SGLangProvider does not support fitting tokens to context")
        if params.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if params.logprobs is not None and params.logprobs < 0:
            raise NotImplementedError("SGLangProvider requires a nonnegative top-logprob count")
        return {
            "max_new_tokens": params.max_tokens,
            "temperature": params.temperature if params.do_sample else 0.0,
            "top_p": params.top_p if params.do_sample and params.top_p is not None else 1.0,
            "top_k": params.top_k if params.do_sample and params.top_k is not None else -1,
            "stop": list(params.stop_sequences or ()),
            "n": 1,
        }

    def _consider_larger_batches(self, outputs: list[LMOutput], params: SamplingParams) -> None:
        """Offer one conservative hint using resolved pools, never raw free VRAM."""
        if self._capacity_checked:
            return
        if params.max_tokens is None or len(outputs) < 16:
            self._previous_batch = None
            return
        prompt_lengths = [output.metadata.get("prompt_tokens") for output in outputs]
        if any(type(length) is not int or length <= 0 for length in prompt_lengths):
            self._previous_batch = None
            return
        if any(output.metadata.get("num_retractions", 0) for output in outputs):
            self._previous_batch = None
            return
        longest_prompt = max(prompt_lengths)
        batch = (len(outputs), params.num_samples, params.max_tokens, longest_prompt)
        previous = self._previous_batch
        self._previous_batch = batch
        if previous is None or previous[:3] != batch[:3]:
            return
        self._capacity_checked = True
        # Telemetry is optional across SGLang versions. A failed diagnostic must
        # never discard successful generations or turn into an inference retry.
        try:
            info = self.engine.get_server_info()
        except Exception:
            logger.debug("SGLang capacity diagnostic unavailable", exc_info=True)
            return
        if not isinstance(info, dict) or info.get("dp_size", 1) != 1:
            return
        states = info.get("internal_states")
        if not isinstance(states, list) or len(states) != 1 or not isinstance(states[0], dict):
            return
        state = states[0]
        # Do not use the requested server_args limit: KDA state pools can lower it.
        capacity = state.get("effective_max_running_requests_per_dp")
        memory = state.get("memory_usage")
        if not isinstance(memory, dict):
            return
        tokens = memory.get("token_capacity")
        if type(capacity) is not int or type(tokens) is not int:
            return
        sequences = len(outputs)
        target = 2 * sequences
        # Reserve half the token pool, using the largest observed expanded prompt
        # and the full output cap, rather than the often much shorter actual output.
        token_budget = target * (max(longest_prompt, previous[3]) + params.max_tokens)
        if capacity < target or tokens < 2 * token_budget:
            return
        chunk_size = target // params.num_samples
        logger.warning(
            "Consider increasing concurrency: repeated SGLang calls supplied %d sequences, "
            "but the effective engine limit is %d and the token-cache capacity is %d. "
            "Try adding '-o batching.chunk_size=%d' after your existing --harness argument. "
            "This supplies up to %d sequences at %d sample(s) per example. "
            "Benchmark throughput and peak memory; this is capacity headroom, not a "
            "guarantee of a speedup or sufficient transient memory. Settings are unchanged.",
            sequences,
            capacity,
            tokens,
            chunk_size,
            target,
            params.num_samples,
        )

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions and retain sampled-token probabilities and engine metadata."""
        params = self._default_sampling_params(sampling_params)
        if self._closed:
            raise RuntimeError("SGLangProvider is closed")
        sampling = self._build_sampling_params(params)
        _require_original_logprobs()
        if not requests:
            return []
        grouped: list[list[LMOutput]] = [[] for _ in requests]
        prepared = []
        positions = []
        first_error: Exception | None = None
        for index, request in enumerate(requests):
            try:
                item = self._prepare_request(request)
            except (ValueError, TypeError, OSError, TemplateError, NotImplementedError) as error:
                first_error = first_error or error
                grouped[index] = [
                    LMOutput(text="", metadata={REQUEST_ERROR_KEY: str(error)})
                    for _ in range(params.num_samples)
                ]
            else:
                prepared.append(item)
                positions.append(index)
        if not prepared:
            assert first_error is not None
            raise first_error
        inputs = {
            key: [item[0][key] for item in prepared for _ in range(params.num_samples)]
            for key in prepared[0][0]
        }
        if "image_data" in inputs and not any(inputs["image_data"]):
            del inputs["image_data"]
        outputs = self.engine.generate(
            **inputs,
            sampling_params=sampling,
            return_logprob=True,
            top_logprobs_num=params.logprobs if params.logprobs is not None else 1,
        )
        if not isinstance(outputs, list) or len(outputs) != len(prepared) * params.num_samples:
            raise RuntimeError("SGLang returned an unexpected number of completions")
        converted = [self._convert_output(output) for output in outputs]
        if all(REQUEST_ERROR_KEY in output.metadata for output in converted):
            raise RuntimeError(
                f"SGLang aborted every generation: {converted[0].metadata[REQUEST_ERROR_KEY]}"
            )
        for offset, position in enumerate(positions):
            samples = converted[offset * params.num_samples : (offset + 1) * params.num_samples]
            failures = [
                o.metadata[REQUEST_ERROR_KEY] for o in samples if REQUEST_ERROR_KEY in o.metadata
            ]
            overflow = bool(failures) and all(
                o.metadata.get(CONTEXT_OVERFLOW_KEY, False)
                for o in samples
                if REQUEST_ERROR_KEY in o.metadata
            )
            for output in samples:
                output.metadata.update(prepared[offset][1])
                if failures:
                    # The scorer expects the complete requested sample budget.
                    # Mark the whole request, including any successful samples,
                    # rather than silently scoring an incomplete pass@k panel.
                    output.metadata[REQUEST_ERROR_KEY] = str(failures[0])
                    if overflow:
                        output.metadata[CONTEXT_OVERFLOW_KEY] = True
            grouped[position] = samples
        self._consider_larger_batches(
            [o for o in converted if REQUEST_ERROR_KEY not in o.metadata], params
        )
        return grouped

    def _convert_output(self, output: dict[str, Any]) -> LMOutput:
        metadata = dict(output["meta_info"])
        finish_reason = metadata.get("finish_reason")
        reason = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
        if reason == "abort":
            # Failed requests are unscored; retain diagnostics, not partial
            # probability arrays in every result-queue payload.
            metadata.pop("output_token_logprobs", None)
            metadata.pop("output_top_logprobs", None)
            message = f"SGLang aborted generation: {finish_reason}"
            metadata.update(
                {
                    REQUEST_ERROR_KEY: message,
                    "finish_reason": "abort",
                    "sglang_finish_reason": finish_reason,
                }
            )
            # Only classify explicit length diagnostics, never generic aborts.
            if any(
                s in message.lower()
                for s in ("context length", "context overflow", "input length", "prompt length")
            ):
                metadata[CONTEXT_OVERFLOW_KEY] = True
            return LMOutput(text=output.get("text", ""), metadata=metadata)
        # The harness counts cap hits from the normalized string, while replay
        # diagnostics still need SGLang's matched token or length detail.
        metadata.update(finish_reason=reason, sglang_finish_reason=finish_reason)
        sampled = metadata.pop("output_token_logprobs")
        if len(sampled) != metadata["completion_tokens"]:
            raise RuntimeError("SGLang sampled logprobs do not cover every generated token")
        top = metadata.pop("output_top_logprobs", [])
        if top and len(top) != len(sampled):
            raise RuntimeError("SGLang top-logprobs do not align with sampled tokens")
        logprobs: list[LogProbEntry] = []
        for index, (logprob, token_id, *token_text) in enumerate(sampled):
            if not math.isfinite(float(logprob)):
                raise RuntimeError("SGLang returned a non-finite sampled-token logprob")
            entry: LogProbEntry = {
                "token": token_text[0]
                if token_text and token_text[0] is not None
                else self.get_tokenizer().decode([token_id]),
                "logprob": float(logprob),
                "token_id": token_id,
            }
            if top:
                entry["top_logprobs"] = [
                    {"token": self.get_tokenizer().decode([item[1]]), "logprob": float(item[0])}
                    for item in top[index]
                ]
            logprobs.append(entry)
        token_ids = [item[1] for item in sampled]
        metadata.update(
            output_token_ids=token_ids,
            sum_logits=sum(entry["logprob"] for entry in logprobs),
            num_tokens=len(logprobs),
            num_tokens_all=metadata["completion_tokens"],
            sglang_return_original_logprob="1",
        )
        return LMOutput(text=output["text"], logprobs=logprobs, metadata=metadata)

    def describe_request(
        self, request: LMRequest, sampling_params: SamplingParams | None = None
    ) -> dict[str, Any]:
        """Record the effective engine request settings."""
        params = self._default_sampling_params(sampling_params)
        self._validate_request(request)
        sampling = self._build_sampling_params(params)
        return {
            "provider": type(self).__name__,
            "mode": "generate",
            "endpoint": "engine.generate",
            "input_mode": "input_ids" if self._add_bos_token is False else "text",
            "generation_kwargs": {
                "max_gen_toks": sampling["max_new_tokens"],
                "temperature": sampling["temperature"],
                "do_sample": params.do_sample and params.temperature > 0,
                "top_p": sampling["top_p"],
                "top_k": sampling["top_k"],
                "num_samples": params.num_samples,
                "logprobs": params.logprobs if params.logprobs is not None else 1,
            },
            "stop_sequences": sampling["stop"],
            "sglang_return_original_logprob": "1",
            "serving_profile": {
                "selected": self.profile_report["selected"],
                "effective_options": dict(self.profile_report["effective_options"]),
            },
        }

    def logprobs(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        """Reject continuation scoring until its token-boundary contract is implemented."""
        raise NotImplementedError("SGLangProvider does not yet support continuation scoring")

    async def agenerate(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        """Run sequential engine calls without blocking the harness event loop."""
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    def close(self) -> None:
        """Release engine processes when the evaluation worker finishes."""
        if not self._closed:
            self.engine.shutdown()
            self._closed = True

    def shutdown(self) -> None:
        """Release engine processes and GPU memory."""
        self.close()
