"""Harness: A model provider configured with specific capabilities.

The Harness wraps an InferenceProvider and applies configuration to all requests.
It provides both single-turn (generate) and multi-turn (run) interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams

from .backends import Backend, get_backend
from .config import HarnessConfig
from .result import HarnessResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider


class Harness:
    """A model provider configured with specific capabilities.

    The Harness wraps an InferenceProvider and applies
    configuration to the requests. It provides both single-turn (generate/agenerate)
    and multi-turn (run) interfaces.

    For multi-turn execution with run(), a backend must be configured.
    """

    def __init__(self, config: HarnessConfig) -> None:
        """Initialize the Harness.

        Args:
            config: Configuration specifying provider, tools, system prompt, etc.
                The provider is created from config.provider.
        """
        self.config = config
        self._provider: InferenceProvider | None = None
        self._backend: Backend | None = None

    @property
    def provider(self) -> InferenceProvider:
        """Get or create the inference provider.

        The provider is lazily created from config.provider on first access.
        If metrics configuration is present and enabled, the provider is wrapped
        with instrumentation for metrics collection.
        """
        if self._provider is None:
            provider = self.config.provider.create_provider()
            if self.config.metrics is not None and self.config.metrics.enabled:
                from olmo_eval.inference.metrics import InstrumentedProvider

                instrumented = InstrumentedProvider(provider)
                if self.config.metrics.collect_gpu:
                    instrumented.enable_gpu_monitoring(interval_s=1.0)
                self._provider = instrumented  # type: ignore[assignment]
            else:
                self._provider = provider
        return self._provider  # type: ignore[return-value]

    @property
    def backend(self) -> Backend:
        """Get or create the backend.

        The backend is lazily created from config.backend on first access.

        Raises:
            RuntimeError: If no backend is configured.
        """
        if self._backend is None:
            if not self.config.backend:
                raise RuntimeError(
                    "No backend configured. Set config.backend to use run(). "
                    "For single-turn generation, use generate() or agenerate() instead."
                )
            self._backend = get_backend(self.config.backend)
        return self._backend

    @property
    def model_name(self) -> str:
        """Get the model name from the provider.

        Returns:
            Model name string.
        """
        return self.provider.model_name

    # ─────────────────────────────────────────────────────────
    # Single-turn interface (same as Provider, but with config)
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Single-turn generation with config injected.

        Applies the harness configuration (tools, system prompt) to each request
        and passes them to the provider.

        Args:
            requests: List of requests to process.
            sampling_params: Optional sampling parameters.

        Returns:
            List of output lists, one per request.
        """
        transformed = [self._apply_config(r) for r in requests]
        return self.provider.generate(transformed, sampling_params)

    def logprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Log probability computation.

        Args:
            requests: List of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return self.provider.logprobs(requests)

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async single-turn generation with config injected."""
        transformed = [self._apply_config(r) for r in requests]
        return await self.provider.agenerate(transformed, sampling_params)

    async def alogprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Async log probability computation."""
        return await self.provider.alogprobs(requests)

    # ─────────────────────────────────────────────────────────
    # Multi-turn interface (delegates to backend)
    # ─────────────────────────────────────────────────────────

    async def run(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> HarnessResult:
        """Multi-turn execution via configured backend.

        Runs an agent loop that:
        1. Sends the request to the model
        2. If the response has tool calls, executes them
        3. Appends results and continues until done or max_turns

        Args:
            request: Initial request to start the conversation.
            sampling_params: Optional sampling parameters.
            trace_metadata: Optional metadata for tracing (e.g., instance_id, task_id).

        Returns:
            HarnessResult with trajectory and final output.

        Raises:
            RuntimeError: If no backend is configured.
        """
        return await self.backend.run(
            self.provider,
            self.config,
            request,
            sampling_params,
            trace_metadata,
            **self.config.backend_kwargs,
        )

    async def cleanup(self) -> None:
        """Clean up resources held by the harness and its backend."""
        self.shutdown_reporters()
        if self._backend is not None:
            await self._backend.cleanup()

    def flush_metrics(self, batch_hash: str, clear: bool = True) -> None:
        """Flush collected metrics to configured reporters.

        Call this after each batch to write metrics incrementally.
        Only has effect if metrics is enabled and the provider is instrumented.

        Args:
            batch_hash: Batch hash computed from native instance IDs.
            clear: If True, clear collected metrics after reporting.
        """
        if self.config.metrics is None or not self.config.metrics.enabled:
            return

        # Check if provider is instrumented
        if self._provider is None:
            return

        from olmo_eval.inference.metrics import InstrumentedProvider

        if not isinstance(self._provider, InstrumentedProvider):
            return

        metrics = self._provider.get_metrics()
        if not metrics:
            return

        # Initialize reporters and report
        from olmo_eval.inference.metrics.core.stats import compute_batch_metrics

        # Get GPU snapshots collected during inference
        gpu_snapshots = self._provider.get_gpu_snapshots()

        batch = compute_batch_metrics(
            metrics,
            wall_clock_s=0.0,
            batch_hash=batch_hash,
            config=self.config.metrics,
            gpu_snapshots=gpu_snapshots,
        )

        # Get or create cached reporters (reuse connections across batches)
        reporters = self._get_reporters()

        for reporter in reporters:
            try:
                reporter.report_batch(batch)
                reporter.flush()
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to report metrics: {e}")

        # Clear metrics after reporting to avoid double-counting
        if clear:
            self._provider.clear_metrics()

    def initialize_reporters(self) -> None:
        """Initialize metrics reporters eagerly.

        Call this at job start to establish database connections early rather than
        waiting until the first batch is processed. This is optional - reporters
        will be lazily initialized on first use if not called.
        """
        self._get_reporters()

    def _get_reporters(self) -> list[Any]:
        """Get or create cached metrics reporters."""
        if not hasattr(self, "_reporters"):
            from olmo_eval.inference.metrics.core.registry import reporter_registry

            self._reporters: list[Any] = []
            if self.config.metrics is not None:
                for reporter_config in self.config.metrics.reporters:
                    resolved = self._resolve_reporter_config(reporter_config)
                    if resolved is not None:
                        reporter = reporter_registry.create(resolved)
                        # Initialize reporters that support eager connection
                        init_fn = getattr(reporter, "initialize", None)
                        if callable(init_fn):
                            init_fn()
                        self._reporters.append(reporter)
        return self._reporters

    def shutdown_reporters(self) -> None:
        """Shutdown cached metrics reporters."""
        import contextlib

        if hasattr(self, "_reporters"):
            for reporter in self._reporters:
                with contextlib.suppress(Exception):
                    reporter.shutdown()
            self._reporters = []

    def _resolve_reporter_config(
        self, reporter_config: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Resolve reporter config, adding path for file reporter if needed."""
        if isinstance(reporter_config, str):
            name = reporter_config
            config_dict: dict[str, Any] = {}
        else:
            name = reporter_config.get("name", "console")
            config_dict = dict(reporter_config)

        # For file reporter, resolve path from metrics config if not set
        if name == "file" and "path" not in config_dict:
            path = self.config.metrics.get_metrics_path() if self.config.metrics else None
            if path is None:
                import logging

                logging.getLogger(__name__).warning(
                    "File reporter requires output_dir in MetricsConfig. Skipping."
                )
                return None
            config_dict["name"] = "file"
            config_dict["path"] = path
            return config_dict

        return reporter_config

    # ─────────────────────────────────────────────────────────
    # Config application (used by backends)
    # ─────────────────────────────────────────────────────────

    def _apply_config(self, request: LMRequest) -> LMRequest:
        """Inject tool schemas and system prompt from config.

        This transforms a request by adding:
        - Tool schemas (if config has tools)
        - System prompt (if configured and not already present)

        Args:
            request: Original request.

        Returns:
            New request with config applied.
        """
        messages = self._inject_system_prompt(request.messages)

        return LMRequest(
            request_type=request.request_type,
            messages=messages,
            prompt=request.prompt,
            continuations=request.continuations,
            tools=self.config.tool_schemas if self.config.has_tools else request.tools,
            system_prompt=self.config.system_prompt or request.system_prompt,
        )

    def _inject_system_prompt(
        self, messages: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        """Add system prompt to messages if configured and not present.

        Args:
            messages: Original message tuple.

        Returns:
            Messages with system prompt prepended if needed.
        """
        if not self.config.system_prompt:
            return messages

        # Check if messages already start with a system message
        if messages and messages[0].get("role") == "system":
            return messages

        # Prepend system message
        system_msg: dict[str, Any] = {
            "role": "system",
            "content": self.config.system_prompt,
        }
        return (system_msg,) + messages


def create_harness(
    config: HarnessConfig | dict[str, Any],
) -> Harness:
    """Create a Harness from config.

    Convenience function that handles config creation/parsing.

    Args:
        config: HarnessConfig instance or dict.

    Returns:
        Configured Harness instance.
    """
    if isinstance(config, dict):
        return Harness(HarnessConfig.from_dict(cast(dict[str, Any], config)))
    return Harness(config)
