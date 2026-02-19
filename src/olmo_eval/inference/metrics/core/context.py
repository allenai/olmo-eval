"""Context manager for scoped metrics collection."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from .collector import InstrumentedHarness, InstrumentedProvider
from .config import MetricsConfig
from .registry import reporter_registry
from .stats import compute_batch_metrics

if TYPE_CHECKING:
    from olmo_eval.harness import Harness
    from olmo_eval.inference.base import InferenceProvider

    from .protocol import MetricsReporter

logger = logging.getLogger(__name__)


class MetricsContext:
    """Context manager for scoped metrics collection.

    Wraps an InferenceProvider or Harness to collect metrics during
    a generation session, then reports them on exit.
    """

    def __init__(
        self,
        target: InferenceProvider | Harness,
        config: MetricsConfig | None = None,
    ) -> None:
        """Initialize the context.

        Args:
            target: InferenceProvider or Harness to instrument.
            config: MetricsConfig with reporters, metadata, and tags.
        """
        self._config = config or MetricsConfig()

        # Determine if target is a Harness or Provider
        # Harness has 'config' attribute, Provider does not
        if hasattr(target, "config") and hasattr(target, "_apply_config"):
            self._collector: InstrumentedProvider | InstrumentedHarness = InstrumentedHarness(
                target  # type: ignore[arg-type]
            )
        else:
            self._collector = InstrumentedProvider(target)

        self._reporters: list[MetricsReporter] = self._init_reporters(list(self._config.reporters))
        self._start_time: float = 0.0

    def _init_reporters(
        self, reporter_configs: list[str | dict[str, Any]]
    ) -> list[MetricsReporter]:
        """Initialize reporters from configs."""
        reporters = []
        for reporter_config in reporter_configs:
            try:
                # Resolve path for jsonl reporter if not explicitly set
                resolved_config = self._resolve_reporter_config(reporter_config)
                if resolved_config is None:
                    # Skip reporter if path can't be resolved
                    continue
                reporter = reporter_registry.create(resolved_config)
                reporters.append(reporter)
            except Exception as e:
                logger.warning(f"Failed to create reporter {reporter_config}: {e}")
        return reporters

    def _resolve_reporter_config(
        self, reporter_config: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Resolve reporter config, adding path for jsonl if needed."""
        # Determine reporter name
        if isinstance(reporter_config, str):
            name = reporter_config
            config_dict: dict[str, Any] = {}
        else:
            name = reporter_config.get("name", "console")
            config_dict = dict(reporter_config)

        # For jsonl reporter, resolve path from metrics config if not set
        if name == "jsonl" and "path" not in config_dict:
            path = self._config.get_metrics_path()
            if path is None:
                logger.warning(
                    "JSONL reporter requires output_dir to be set in MetricsConfig. "
                    "Skipping jsonl reporter."
                )
                return None
            config_dict["name"] = "jsonl"
            config_dict["path"] = path
            return config_dict

        return reporter_config

    def __enter__(self) -> InstrumentedProvider | InstrumentedHarness:
        """Enter the context and return the instrumented target."""
        self._start_time = time.perf_counter()
        return self._collector

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and report metrics."""
        wall_clock = time.perf_counter() - self._start_time
        metrics = self._collector.get_metrics()

        if metrics:
            batch = compute_batch_metrics(metrics, wall_clock, self._config)
            for reporter in self._reporters:
                try:
                    reporter.report_batch(batch)
                    reporter.flush()
                except Exception as e:
                    # Reporter failures should not crash the evaluation
                    logger.warning(f"Reporter {reporter.reporter_name} failed: {e}")
                finally:
                    with contextlib.suppress(Exception):
                        reporter.shutdown()


def collect_metrics(
    target: InferenceProvider | Harness,
    config: MetricsConfig | None = None,
    *,
    reporters: list[str | dict[str, Any]] | None = None,
    tags: dict[str, str] | None = None,
) -> MetricsContext:
    """Context manager for metrics collection.

    Args:
        target: InferenceProvider or Harness to instrument.
        config: Full MetricsConfig (preferred). If provided, other kwargs are ignored.
        reporters: List of reporter names or configs (default: ["console"]).
            Supported reporters: "console", "jsonl".
            Dict format: {"name": "jsonl", "path": "/path/to/file.jsonl"}
        tags: User-defined tags to attach to metrics.

    Returns:
        Context manager that yields instrumented target.

    Example:
        >>> provider = create_provider("vllm", model_name="llama-3.1-8b")
        >>> with collect_metrics(provider, reporters=["console"]) as ctx:
        ...     outputs = ctx.generate(requests, sampling_params)

        >>> # Or with full config:
        >>> config = MetricsConfig(
        ...     reporters=("console",),
        ...     experiment_id="exp-123",
        ...     model_name="llama-3.1-8b",
        ... )
        >>> with collect_metrics(provider, config=config) as ctx:
        ...     outputs = ctx.generate(requests, sampling_params)
    """
    if config is None:
        # Build config from kwargs for convenience
        config = MetricsConfig(
            reporters=tuple(reporters) if reporters else ("console",),
            tags=tags or {},
        )
    return MetricsContext(target, config)
