"""Inference metrics collection and reporting.

This module provides tools for collecting performance metrics during inference:
- Latency, throughput, token counts
- Time to first token (TTFT) when available
- GPU utilization (optional, requires pynvml)
"""

from .core.collector import InstrumentedHarness, InstrumentedProvider
from .core.config import MetricsConfig, ReporterType
from .core.gpu import GPUMonitor, collect_gpu_snapshots, is_gpu_available
from .core.registry import reporter_registry
from .core.schema import BatchMetrics, GPUSnapshot, RequestMetrics
from .core.stats import compute_batch_hash

__all__ = [
    "MetricsConfig",
    "ReporterType",
    "RequestMetrics",
    "BatchMetrics",
    "GPUSnapshot",
    "GPUMonitor",
    "collect_gpu_snapshots",
    "is_gpu_available",
    "InstrumentedProvider",
    "InstrumentedHarness",
    "reporter_registry",
    "compute_batch_hash",
]
