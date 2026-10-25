"""Inline SGLang provider for text and chat generation."""

from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Any

from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider


class SGLangProvider(InferenceProvider):
    """Run generation with a locally installed SGLang engine.

    Install the engine and any model registration package separately. Continuation
    likelihood scoring and tool requests are not supported by this provider.
    """

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        max_model_len: int | None = None,
        max_concurrency: int | None = None,
        add_bos_token: bool | None = None,
        model_registration: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name)
        if kwargs.get("skip_tokenizer_init"):
            raise ValueError("SGLangProvider requires tokenizer initialization for chat and stops")
        if model_registration:
            module, name = model_registration.split(":", 1)
            getattr(import_module(module), name)()
        options = dict(
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
        self._add_bos_token = add_bos_token
        self.engine = import_module("sglang").Engine(**options)

    def get_tokenizer(self) -> Any:
        """Return the engine's tokenizer."""
        return self.engine.tokenizer_manager.tokenizer

    def _format_prompt(self, request: LMRequest) -> str:
        if request.tools or request.request_type == RequestType.LOGLIKELIHOOD:
            raise NotImplementedError("SGLangProvider supports text and chat generation only")
        if request.request_type == RequestType.CHAT:
            if not request.messages:
                raise ValueError("CHAT requests require messages")
            return self.get_tokenizer().apply_chat_template(
                list(request.messages), tokenize=False, add_generation_prompt=True
            )
        return request.prompt

    def _build_sampling_params(self, params: SamplingParams) -> dict[str, Any]:
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            raise NotImplementedError("SGLangProvider does not support prompt truncation")
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

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions and retain sampled-token probabilities and engine metadata."""
        params = self._default_sampling_params(sampling_params)
        sampling = self._build_sampling_params(params)
        if not requests:
            return []
        prompts = [self._format_prompt(req) for req in requests for _ in range(params.num_samples)]
        if self._add_bos_token is False:
            inputs = {
                "input_ids": [
                    self.get_tokenizer().encode(p, add_special_tokens=False) for p in prompts
                ]
            }
        else:
            inputs = {"prompt": prompts}
        outputs = self.engine.generate(
            **inputs,
            sampling_params=sampling,
            return_logprob=True,
            top_logprobs_num=params.logprobs if params.logprobs is not None else 1,
        )
        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            raise RuntimeError("SGLang returned an unexpected number of completions")
        converted = [self._convert_output(output) for output in outputs]
        return [
            converted[start : start + params.num_samples]
            for start in range(0, len(converted), params.num_samples)
        ]

    def _convert_output(self, output: dict[str, Any]) -> LMOutput:
        metadata = dict(output["meta_info"])
        sampled = metadata["output_token_logprobs"]
        top = metadata.get("output_top_logprobs", [])
        if top and len(top) != len(sampled):
            raise RuntimeError("SGLang top-logprobs do not align with sampled tokens")
        logprobs: list[LogProbEntry] = []
        for index, (logprob, token_id, *token_text) in enumerate(sampled):
            entry: LogProbEntry = {
                "token": token_text[0]
                if token_text and token_text[0] is not None
                else self.get_tokenizer().decode([token_id]),
                "logprob": float(logprob),
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
        )
        return LMOutput(text=output["text"], logprobs=logprobs, metadata=metadata)

    def describe_request(
        self, request: LMRequest, sampling_params: SamplingParams | None = None
    ) -> dict[str, Any]:
        """Record the effective engine request settings."""
        params = self._default_sampling_params(sampling_params)
        self._format_prompt(request)
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

    def shutdown(self) -> None:
        """Release engine processes and GPU memory."""
        self.engine.shutdown()
