"""Inference metrics collection and reporting.

This module provides tools for collecting performance metrics during inference:
- Latency, throughput, token counts
- Time to first token (TTFT) when available
- GPU utilization (optional, requires pynvml)

Usage:
    from olmo_eval.inference import create_provider
    from olmo_eval.inference.metrics import collect_metrics

    provider = create_provider("vllm", model_name="llama-3.1-8b")

    with collect_metrics(provider, reporters=["console"]) as ctx:
        outputs = ctx.generate(requests, sampling_params)
"""

from .core.collector import InstrumentedHarness, InstrumentedProvider
from .core.config import MetricsConfig, ReporterName
from .core.context import MetricsContext, collect_metrics
from .core.registry import reporter_registry
from .core.schema import BatchMetrics, GPUSnapshot, RequestMetrics

__all__ = [
    # Main API
    "collect_metrics",
    "MetricsContext",
    "MetricsConfig",
    "ReporterName",
    # Schema
    "RequestMetrics",
    "BatchMetrics",
    "GPUSnapshot",
    # Advanced
    "InstrumentedProvider",
    "InstrumentedHarness",
    "reporter_registry",
]
