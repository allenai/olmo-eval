"""vLLM backend."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from olmo_eval.core import LMOutput, LMRequest, SamplingParams

from .base import Backend

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput


def _get_token_string(logprob_obj: Any, token_id: int, tokenizer: Any = None) -> str:
    """Extract token string from vLLM logprob object."""
    if hasattr(logprob_obj, "decoded_token"):
        return logprob_obj.decoded_token
    if tokenizer is not None:
        return tokenizer.decode([token_id])
    return str(token_id)


class VLLMBackend(Backend):
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
            raise ImportError("vllm is required for VLLMBackend") from e

        super().__init__(model_name)
        engine_kwargs.setdefault("gpu_memory_utilization", 0.7)
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
        params = self._default_sampling_params(sampling_params)
        vllm_params = self._build_sampling_params(params)

        prompts = [req.prompt for req in requests]
        outputs: list[RequestOutput] = self.llm.generate(prompts, vllm_params)

        return [
            [
                LMOutput(
                    text=completion.text,
                    logprobs=self._convert_logprobs(completion.logprobs),
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

                for token_id, token_probs in zip(cont_tokens, cont_logprobs, strict=False):
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


class AsyncVLLMBackend:
    """Async vLLM backend with continuous batching for streaming results.

    Uses vLLM's AsyncLLMEngine to enable true continuous batching where
    requests can be added while others are processing, and results stream
    back as they complete.
    """

    def __init__(self, model_name: str, **engine_kwargs) -> None:
        """Initialize the async backend.

        Args:
            model_name: HuggingFace model identifier or local path.
            **engine_kwargs: Additional arguments passed to vLLM engine.
        """
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        try:
            from vllm import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
        except ImportError as e:
            raise ImportError("vllm is required for AsyncVLLMBackend") from e

        self.model_name = model_name
        engine_kwargs.setdefault("gpu_memory_utilization", 0.7)

        engine_args = AsyncEngineArgs(model=model_name, **engine_kwargs)
        self.engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)
        self._request_counter = 0

    def _get_next_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"req-{self._request_counter}"

    async def add_request(
        self,
        request_id: str,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """Add a single request to the engine (non-blocking).

        Args:
            request_id: Unique identifier for this request.
            request: The LM request to process.
            sampling_params: Optional sampling parameters.
        """
        from vllm import SamplingParams as VLLMSamplingParams

        params = sampling_params or SamplingParams()
        vllm_params = VLLMSamplingParams(
            max_tokens=params.max_tokens,
            n=params.num_samples,
            temperature=params.temperature if params.temperature > 0 else 0.0,
            top_p=params.top_p if params.top_p else 1.0,
            top_k=params.top_k if params.top_k else -1,
            stop=list(params.stop_sequences) if params.stop_sequences else None,
            logprobs=params.logprobs,
        )

        await self.engine.add_request(
            request_id=request_id,
            prompt=request.prompt,
            params=vllm_params,
        )

    async def stream_results(self) -> AsyncIterator[tuple[str, list[LMOutput]]]:
        """Stream results as they complete.

        Yields:
            Tuples of (request_id, list of LMOutput for that request).
        """
        async for output in self.engine.generate(None, None):
            if output.finished:
                outputs = [
                    LMOutput(
                        text=completion.text,
                        logprobs=self._convert_logprobs(completion.logprobs),
                    )
                    for completion in output.outputs
                ]
                yield output.request_id, outputs

    def _convert_logprobs(self, vllm_logprobs: list | None) -> list[dict] | None:
        """Convert vLLM logprobs format to standard format."""
        if vllm_logprobs is None:
            return None

        result = []
        for token_logprobs in vllm_logprobs:
            if not token_logprobs:
                continue
            token_id, logprob_obj = next(iter(token_logprobs.items()))
            token_str = _get_token_string(logprob_obj, token_id)
            result.append({"token": token_str, "logprob": logprob_obj.logprob})

        return result

    async def shutdown(self) -> None:
        """Shutdown the engine gracefully."""
        if hasattr(self.engine, "shutdown"):
            await self.engine.shutdown()
