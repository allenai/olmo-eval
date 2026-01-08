"""vLLM backend for high-performance inference."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from olmo_eval.core import LMOutput, LMRequest, SamplingParams

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.outputs import RequestOutput


def _get_token_string(logprob_obj: Any, token_id: int, tokenizer: Any = None) -> str:
    """Extract token string from vLLM logprob object."""
    if hasattr(logprob_obj, "decoded_token"):
        return logprob_obj.decoded_token
    if tokenizer is not None:
        return tokenizer.decode([token_id])
    return str(token_id)


class VLLMBackend:
    """Backend using vLLM for high-throughput inference."""

    def __init__(self, model_name: str, **engine_kwargs) -> None:
        """Initialize the backend.

        Args:
            model_name: HuggingFace model identifier or local path.
            **engine_kwargs: Additional arguments passed to vLLM LLM engine.
        """
        # Suppress verbose vLLM logging
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vllm is required for VLLMBackend. Install with: pip install vllm"
            ) from e

        engine_kwargs.setdefault("gpu_memory_utilization", 0.7)
        self.model_name = model_name
        self.llm: LLM = LLM(model=model_name, **engine_kwargs)

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

    def _convert_logprobs(self, vllm_logprobs: list | None) -> list[dict] | None:
        """Convert vLLM logprobs format to standard format."""
        if vllm_logprobs is None:
            return None

        result = []
        for token_logprobs in vllm_logprobs:
            if not token_logprobs:
                continue
            # vLLM returns dict of {token_id: LogprobInfo}, take first (chosen) token
            token_id, logprob_obj = next(iter(token_logprobs.items()))
            token_str = _get_token_string(logprob_obj, token_id)
            result.append({"token": token_str, "logprob": logprob_obj.logprob})

        return result

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = sampling_params or SamplingParams()
        vllm_params = self._build_sampling_params(params)

        prompts = [req.prompt for req in requests]
        outputs: list[RequestOutput] = self.llm.generate(prompts, vllm_params)

        return [
            [
                LMOutput(text=completion.text, logprobs=self._convert_logprobs(completion.logprobs))
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

        # Build full prompts for all continuations
        full_prompts = [
            request.prompt + continuation
            for request in requests
            for continuation in (request.continuations or ())
        ]

        outputs: list[RequestOutput] = self.llm.generate(full_prompts, vllm_params)
        output_iter = iter(outputs)
        tokenizer = self.llm.get_tokenizer()

        # Parse results back to per-request structure
        results = []
        for request in requests:
            continuations = request.continuations or ()
            ctx_len = len(tokenizer.encode(request.prompt, add_special_tokens=False))
            request_outputs = []

            for continuation in continuations:
                output = next(output_iter)
                full_tokens = tokenizer.encode(
                    request.prompt + continuation, add_special_tokens=False
                )
                cont_tokens = full_tokens[ctx_len:]

                logprob_entries = []
                total = 0.0

                prompt_logprobs = output.prompt_logprobs or []
                cont_logprobs = prompt_logprobs[ctx_len:]

                for token_id, token_probs in zip(cont_tokens, cont_logprobs):
                    if token_probs and token_id in token_probs:
                        lp_obj = token_probs[token_id]
                        token_str = _get_token_string(lp_obj, token_id, tokenizer)
                        logprob_entries.append({"token": token_str, "logprob": lp_obj.logprob})
                        total += lp_obj.logprob

                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=logprob_entries,
                        metadata={"total_logprob": total},
                    )
                )

            results.append(request_outputs)

        return results
