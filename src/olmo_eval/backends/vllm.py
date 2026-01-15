"""vLLM backend."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from olmo_eval.core import LMOutput, LMRequest, SamplingParams

from .base import Backend

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

    Uses vLLM's AsyncLLM (V1 engine) or AsyncLLMEngine (legacy) to enable
    true continuous batching where requests can be added while others are
    processing, and results stream back as they complete.
    """

    def __init__(self, model_name: str, **engine_kwargs) -> None:
        """Initialize the async backend.

        Args:
            model_name: HuggingFace model identifier or local path.
            **engine_kwargs: Additional arguments passed to vLLM engine.
        """
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        self.model_name = model_name
        engine_kwargs.setdefault("gpu_memory_utilization", 0.7)

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
                raise ImportError("vllm is required for AsyncVLLMBackend") from e

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
        """
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
                            logprobs=self._convert_logprobs(completion.logprobs),
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
