"""Experiment plan data structure for Beaker launch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from olmo_eval.core.types import RunnerType

if TYPE_CHECKING:
    from olmo_eval.launch import BeakerModelSpec


@dataclass
class ExperimentPlan:
    """A single experiment to be launched on Beaker.

    This represents the configuration for one Beaker experiment, which may
    contain one or more models running the same set of tasks.

    Attributes:
        name: Experiment name for Beaker.
        model_cfgs: List of BeakerModelSpec objects for models in this experiment.
        model_specs: List of model specification strings (names/paths).
        priority: Job priority level (low, normal, high, urgent).
        tasks: List of task specs to run.
        original_task_specs: Original task specs from config (before expansion).
        total_expanded_tasks: Number of tasks after expansion.
        model_gpu_counts: GPU count for each model (parallel to model_cfgs).
        num_gpus: Total GPUs needed for this experiment.
        parallelism: Number of parallel instances (for parallelism mode).
        split_index: If experiment was split, the 1-based index of this split.
        total_splits: If experiment was split, total number of splits.
        runner_type: The runner type for this experiment (agent tasks force AGENT).
    """

    name: str
    model_cfgs: list[BeakerModelSpec]
    model_specs: list[str]
    priority: str
    tasks: list[str]
    original_task_specs: list[str]
    total_expanded_tasks: int
    model_gpu_counts: list[int]
    num_gpus: int
    parallelism: int = 1
    split_index: int | None = None
    total_splits: int | None = None
    runner_type: RunnerType = RunnerType.SYNC

    # Per-model overrides to pass via -o flags (parallel to model_cfgs)
    # Each entry is a list of override strings like ["gpus=4", "load_format=auto"]
    model_overrides: list[list[str]] = field(default_factory=list)

    # Per-task overrides to pass via -o flags (task_spec -> [overrides])
    # Maps task spec to list of override strings like ["limit=100", "priority=urgent"]
    task_overrides: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure model_overrides is populated."""
        if not self.model_overrides:
            self.model_overrides = [[] for _ in self.model_cfgs]
