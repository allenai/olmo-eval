"""Experiment plan data structure for Beaker launch."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentPlan:
    """A single experiment to be launched on Beaker."""

    name: str
    model_spec: str
    priority: str
    tasks: list[str]
    original_task_specs: list[str]
    total_expanded_tasks: int
    num_gpus: int
    parallelism: int = 1
    split_index: int | None = None
    total_splits: int | None = None

    task_overrides: dict[str, list[str]] = field(default_factory=dict)
