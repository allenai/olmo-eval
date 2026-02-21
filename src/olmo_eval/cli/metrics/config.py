"""Configuration and data models for metrics plotting."""

from __future__ import annotations

from dataclasses import dataclass

# Metrics configuration: key -> (db_path, plot_name, table_name)
METRICS = {
    "throughput": ("output_tokens_per_second", "Tokens/sec", "TPS"),
    "latency": ("mean_latency_s", "Latency", "Latency"),
    "gpu_util": ("metadata_.gpu_summary.avg_utilization_pct", "GPU %", "GPU %"),
    "gpu_mem": ("metadata_.gpu_summary.avg_memory_used_mb", "GPU MB", "GPU MB"),
}

# Metrics that show p95 in the stats table
P95_METRICS = {"throughput", "gpu_util"}

METRICS_DB_NAME = "olmo_eval_metrics"
SERIES_COLORS = ["#5cb8ff", "#ff7f50", "#3cb371", "#9370db", "#ffd700"]


@dataclass(frozen=True)
class MetricInfo:
    """Metadata for a metric."""

    key: str
    path: str
    plot_name: str
    table_name: str


@dataclass
class QueryFilters:
    """Filter parameters for querying samples."""

    experiment_ids: tuple[str, ...]
    experiment_groups: tuple[str, ...]
    model_names: tuple[str, ...]
    model_hashes: tuple[str, ...]
    task_names: tuple[str, ...]
    task_hashes: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_ids": self.experiment_ids,
            "experiment_groups": self.experiment_groups,
            "model_names": self.model_names,
            "model_hashes": self.model_hashes,
            "task_names": self.task_names,
            "task_hashes": self.task_hashes,
        }


@dataclass
class DbConfig:
    """Database connection configuration."""

    host: str
    port: int
    user: str
    password: str
