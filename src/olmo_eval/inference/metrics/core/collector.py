"""Instrumented wrappers for providers and harnesses."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from .gpu import GPUMonitor
from .schema import GPUSnapshot, RequestMetrics
from .timer import Timer

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
    from olmo_eval.harness import Harness
    from olmo_eval.inference.base import InferenceProvider


class InstrumentedChatCompletions:
    """Wraps AsyncOpenAI chat.completions to collect metrics."""

    def __init__(
        self,
        completions: Any,
        metrics_collector: InstrumentedProvider,
    ) -> None:
        self._completions = completions
        self._collector = metrics_collector

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    async def create(self, **kwargs: Any) -> Any:
        """Wrap create() to collect timing metrics."""
        self._collector._start_gpu_monitor_if_needed()

        with Timer() as t:
            response = await self._completions.create(**kwargs)

        # Extract metrics from response
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # Get finish reason from first choice
        choices = getattr(response, "choices", [])
        finish_reason = None
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)

        # Compute tokens per second
        tps = completion_tokens / t.elapsed_s if t.elapsed_s > 0 else 0.0

        metrics = RequestMetrics(
            request_id=str(uuid.uuid4()),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            end_to_end_latency_s=t.elapsed_s,
            tokens_per_second=tps,
            model=kwargs.get("model", self._collector.model_name),
            finish_reason=finish_reason,
        )
        self._collector._request_metrics.append(metrics)

        return response


class InstrumentedChat:
    """Wraps AsyncOpenAI chat to provide instrumented completions."""

    def __init__(self, chat: Any, metrics_collector: InstrumentedProvider) -> None:
        self._chat = chat
        self._collector = metrics_collector
        self._completions: InstrumentedChatCompletions | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)

    @property
    def completions(self) -> InstrumentedChatCompletions:
        if self._completions is None:
            self._completions = InstrumentedChatCompletions(self._chat.completions, self._collector)
        return self._completions


class InstrumentedAsyncOpenAI:
    """Wraps AsyncOpenAI client to collect metrics on chat completions."""

    def __init__(self, client: AsyncOpenAI, metrics_collector: InstrumentedProvider) -> None:
        self._client = client
        self._collector = metrics_collector
        self._chat: InstrumentedChat | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def chat(self) -> InstrumentedChat:
        if self._chat is None:
            self._chat = InstrumentedChat(self._client.chat, self._collector)
        return self._chat


class InstrumentedProvider:
    """Wraps InferenceProvider to collect timing metrics.

    This wrapper intercepts generate/agenerate calls to measure latency
    and collect token counts. All other attributes are forwarded to the
    underlying provider.

    GPU monitoring can be enabled via enable_gpu_monitoring(). When enabled,
    the monitor samples GPU metrics in a background thread during inference.
    """

    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider
        self._request_metrics: list[RequestMetrics] = []
        self._gpu_monitor: GPUMonitor | None = None
        self._gpu_monitoring_enabled = False

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the underlying provider."""
        return getattr(self._provider, name)

    def enable_gpu_monitoring(self, interval_s: float = 1.0) -> None:
        """Enable GPU metrics collection during inference.

        Args:
            interval_s: Sampling interval in seconds.
        """
        self._gpu_monitoring_enabled = True
        self._gpu_monitor = GPUMonitor(interval_s=interval_s)

    def disable_gpu_monitoring(self) -> None:
        """Disable GPU metrics collection."""
        self._gpu_monitoring_enabled = False
        if self._gpu_monitor is not None:
            self._gpu_monitor.stop()
            self._gpu_monitor = None

    def get_openai_client(self) -> InstrumentedAsyncOpenAI:
        """Get an instrumented AsyncOpenAI client.

        Returns an instrumented wrapper that collects metrics on chat completions.
        This allows backends using the OpenAI client directly to still have their
        API calls instrumented.
        """
        client = self._provider.get_openai_client()
        return InstrumentedAsyncOpenAI(client, self)

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate with timing instrumentation."""
        self._start_gpu_monitor_if_needed()

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
        self._start_gpu_monitor_if_needed()

        with Timer() as t:
            outputs = await self._provider.agenerate(requests, sampling_params)

        self._collect_metrics(requests, outputs, t.elapsed_s)
        return outputs

    def logprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Log probability computation with timing instrumentation."""
        self._start_gpu_monitor_if_needed()

        with Timer() as t:
            outputs = self._provider.logprobs(requests)

        self._collect_metrics(requests, outputs, t.elapsed_s)
        return outputs

    async def alogprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Async log probability computation with timing instrumentation."""
        self._start_gpu_monitor_if_needed()

        with Timer() as t:
            outputs = await self._provider.alogprobs(requests)

        self._collect_metrics(requests, outputs, t.elapsed_s)
        return outputs

    def get_metrics(self) -> list[RequestMetrics]:
        """Get collected request metrics."""
        return list(self._request_metrics)

    def get_gpu_snapshots(self) -> tuple[GPUSnapshot, ...]:
        """Stop GPU monitor and return collected snapshots.

        Returns:
            Tuple of GPU snapshots collected since monitoring started.
        """
        if self._gpu_monitor is None:
            return ()
        return self._gpu_monitor.stop()

    def clear_metrics(self) -> None:
        """Clear collected metrics and restart GPU monitor if enabled."""
        self._request_metrics.clear()
        # Stop current monitor to clear its snapshots
        if self._gpu_monitor is not None:
            self._gpu_monitor.stop()
            # Create fresh monitor for next batch
            self._gpu_monitor = GPUMonitor(interval_s=self._gpu_monitor._interval_s)

    def _start_gpu_monitor_if_needed(self) -> None:
        """Start GPU monitor if enabled and not already running."""
        if (
            self._gpu_monitoring_enabled
            and self._gpu_monitor is not None
            and self._gpu_monitor._thread is None
        ):
            self._gpu_monitor.start()

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
        # Try to get accurate token counts from output metadata (set by provider)
        prompt_tokens = 0
        completion_tokens = 0

        for out in outputs:
            if out.metadata:
                # Use server-provided counts if available (most accurate)
                if "prompt_tokens" in out.metadata:
                    prompt_tokens = out.metadata["prompt_tokens"]
                if "completion_tokens" in out.metadata:
                    completion_tokens += out.metadata["completion_tokens"]

        # Fall back to estimation if metadata not available
        if prompt_tokens == 0:
            prompt_tokens = self._count_prompt_tokens(request)
        if completion_tokens == 0:
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
