"""Configuration and data models for metrics plotting."""

from __future__ import annotations

from dataclasses import dataclass

# Metrics configuration: key -> (db_path, plot_name, table_name)
METRICS = {
    "throughput": ("output_tokens_per_second", "Tokens per Second", "TPS"),
    "latency": ("mean_latency_s", "Request Latency", "Latency"),
    "gpu_util": ("metadata_.gpu_summary.avg_utilization_pct", "GPU %", "GPU %"),
    "gpu_mem": ("metadata_.gpu_summary.avg_memory_used_mb", "GPU MB", "GPU MB"),
}

# Metrics that show p95 in the stats table
P95_METRICS = {"throughput", "gpu_util"}

METRICS_DB_NAME = "olmo_eval_metrics"

# Rich/hex colors for plot series - expanded palette for many runs
# Interleaved by hue for maximum distinction between adjacent series
SERIES_COLORS = [
    "#5cb8ff",  # Sky blue
    "#ff7f50",  # Coral
    "#3cb371",  # Medium sea green
    "#9370db",  # Medium purple
    "#ffd700",  # Gold
    "#00ced1",  # Dark turquoise
    "#ff69b4",  # Hot pink
    "#32cd32",  # Lime green
    "#4169e1",  # Royal blue
    "#ffa500",  # Orange
    "#8a2be2",  # Blue violet
    "#40e0d0",  # Turquoise
    "#2e8b57",  # Sea green
    "#ba55d3",  # Medium orchid
    "#ff8c00",  # Dark orange
    "#00bfff",  # Deep sky blue
    "#00fa9a",  # Medium spring green
    "#da70d6",  # Orchid
    "#daa520",  # Goldenrod
    "#1e90ff",  # Dodger blue
    "#ee82ee",  # Violet
    "#66cdaa",  # Medium aquamarine
    "#ffb347",  # Pastel orange
    "#9932cc",  # Dark orchid
    "#48d1cc",  # Medium turquoise
    "#98fb98",  # Pale green
    "#6495ed",  # Cornflower blue
    "#f0e68c",  # Khaki
    "#dda0dd",  # Plum
    "#20b2aa",  # Light sea green
    "#87ceeb",  # Light sky blue
    "#90ee90",  # Light green
    "#cd853f",  # Peru
    "#7fffd4",  # Aquamarine
    "#bdb76b",  # Dark khaki
    "#5f9ea0",  # Cadet blue
    "#d2691e",  # Chocolate
    "#afeeee",  # Pale turquoise
    "#b8860b",  # Dark goldenrod
    "#008b8b",  # Dark cyan
    "#a0522d",  # Sienna
    "#00ffff",  # Cyan
    "#bc8f8f",  # Rosy brown
    "#008080",  # Teal
    "#8b4513",  # Saddle brown
    "#778899",  # Light slate gray
    "#708090",  # Slate gray
    "#2f4f4f",  # Dark slate gray
]


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
