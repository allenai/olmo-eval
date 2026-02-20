"""Instrumented wrappers for providers and harnesses."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from .schema import RequestMetrics
from .timer import Timer

if TYPE_CHECKING:
    from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
    from olmo_eval.harness import Harness
    from olmo_eval.inference.base import InferenceProvider


class InstrumentedProvider:
    """Wraps InferenceProvider to collect timing metrics.

    This wrapper intercepts generate/agenerate calls to measure latency
    and collect token counts. All other attributes are forwarded to the
    underlying provider.
    """

    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider
        self._request_metrics: list[RequestMetrics] = []

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the underlying provider."""
        return getattr(self._provider, name)

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate with timing instrumentation."""
        with Timer() as t:
            outputs = self._provider.generate(requests, sampling_params)

        self._collect_metrics(requests, outputs, t.elapsed_s)
        return outputs

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate with timing instrumentation."""
        with Timer() as t:
            outputs = await self._provider.agenerate(requests, sampling_params)

        self._collect_metrics(requests, outputs, t.elapsed_s)
        return outputs

    def get_metrics(self) -> list[RequestMetrics]:
        """Get collected metrics."""
        return list(self._request_metrics)

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self._request_metrics.clear()

    def _collect_metrics(
        self,
        requests: list[LMRequest],
        outputs: list[list[LMOutput]],
        total_latency_s: float,
    ) -> None:
        """Build metrics from requests and outputs."""
        num_requests = len(requests)
        if num_requests == 0:
            return

        # Distribute latency evenly across requests (best we can do without streaming)
        per_request_latency = total_latency_s / num_requests

        for req, out_list in zip(requests, outputs, strict=True):
            metrics = self._build_request_metrics(req, out_list, per_request_latency)
            self._request_metrics.append(metrics)

    def _build_request_metrics(
        self,
        request: LMRequest,
        outputs: list[LMOutput],
        latency_s: float,
    ) -> RequestMetrics:
        """Build RequestMetrics from a single request/output pair."""
        # Count prompt tokens
        prompt_tokens = self._count_prompt_tokens(request)

        # Count completion tokens across all outputs
        completion_tokens = sum(self._count_output_tokens(out) for out in outputs)

        # Compute tokens per second
        tps = completion_tokens / latency_s if latency_s > 0 else 0.0

        # Get finish reason from first output if available
        finish_reason = None
        if outputs and outputs[0].metadata:
            finish_reason = outputs[0].metadata.get("finish_reason")

        return RequestMetrics(
            request_id=str(uuid.uuid4()),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            end_to_end_latency_s=latency_s,
            tokens_per_second=tps,
            model=self.model_name,
            finish_reason=finish_reason,
        )

    def _count_prompt_tokens(self, request: LMRequest) -> int:
        """Estimate prompt token count."""
        # Try to get tokenizer for accurate count
        try:
            tokenizer = self.get_tokenizer()
            if tokenizer is not None:
                if request.prompt:
                    return len(tokenizer.encode(request.prompt))
                elif request.messages:
                    # Concatenate message content for rough estimate
                    text = " ".join(
                        m.get("content", "")
                        for m in request.messages
                        if isinstance(m.get("content"), str)
                    )
                    return len(tokenizer.encode(text))
        except Exception:
            pass

        # Fall back to word-based estimate (rough approximation)
        if request.prompt:
            return len(request.prompt.split()) * 4 // 3  # ~1.3 tokens per word
        elif request.messages:
            text = " ".join(
                m.get("content", "") for m in request.messages if isinstance(m.get("content"), str)
            )
            return len(text.split()) * 4 // 3
        return 0

    def _count_output_tokens(self, output: LMOutput) -> int:
        """Estimate output token count."""
        # If logprobs are available, use their length
        if output.logprobs:
            return len(output.logprobs)

        # Try to get tokenizer for accurate count
        try:
            tokenizer = self.get_tokenizer()
            if tokenizer is not None and output.text:
                return len(tokenizer.encode(output.text))
        except Exception:
            pass

        # Fall back to word-based estimate
        if output.text:
            return len(output.text.split()) * 4 // 3
        return 0


class InstrumentedHarness:
    """Wraps Harness to collect metrics on provider calls.

    This wrapper intercepts the harness's generate/agenerate calls
    and instruments the underlying provider. All other attributes
    are forwarded to the underlying harness.
    """

    def __init__(self, harness: Harness) -> None:
        self._harness = harness
        self._instrumented_provider: InstrumentedProvider | None = None

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the underlying harness."""
        return getattr(self._harness, name)

    @property
    def _provider(self) -> InstrumentedProvider:
        """Get or create instrumented provider."""
        if self._instrumented_provider is None:
            self._instrumented_provider = InstrumentedProvider(self._harness.provider)
        return self._instrumented_provider

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate with timing instrumentation."""
        transformed = [self._harness._apply_config(r) for r in requests]
        return self._provider.generate(transformed, sampling_params)

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate with timing instrumentation."""
        transformed = [self._harness._apply_config(r) for r in requests]
        return await self._provider.agenerate(transformed, sampling_params)

    def get_metrics(self) -> list[RequestMetrics]:
        """Get collected metrics."""
        return self._provider.get_metrics()

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self._provider.clear_metrics()
