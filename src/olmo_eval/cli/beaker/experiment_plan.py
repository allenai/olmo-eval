"""Experiment plan data structure for Beaker launch."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentPlan:
    """A single experiment to be launched on Beaker."""

    name: str
    model_specs: list[str]
    priority: str
    tasks: list[str]
    original_task_specs: list[str]
    total_expanded_tasks: int
    num_gpus: int
    parallelism: int = 1
    split_index: int | None = None
    total_splits: int | None = None

    model_overrides: list[list[str]] = field(default_factory=list)
    task_overrides: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_overrides:
            self.model_overrides = [[] for _ in self.model_specs]
