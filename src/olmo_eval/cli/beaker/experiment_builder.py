"""Experiment plan building for Beaker launch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig


class ExperimentPlanBuilder:
    """Builds experiment plans with task splits."""

    def __init__(
        self,
        config: LaunchConfig,
        tasks_by_priority: dict[str, list[str]],
        override_priority: str | None = None,
    ):
        self.config = config
        self.tasks_by_priority = tasks_by_priority
        self.override_priority = override_priority

    def _build_model_overrides(self, m_spec: str, original_index: int) -> list[str]:
        """Build list of -o override strings for a model."""
        overrides: list[str] = []

        if original_index < len(self.config.model_overrides):
            overrides.extend(self.config.model_overrides[original_index])

        return overrides

    def _build_experiments(
        self,
        model_specs: list[str],
        model_indices: list[int],
        tasks: list[str],
        priority: str,
        total_expanded_tasks: int,
        multiple_models: bool,
        multiple_priorities: bool,
        task_overrides: dict[str, list[str]],
    ) -> list[ExperimentPlan]:
        """Build one experiment per model."""
        from olmo_eval.launch import get_model_short_name

        experiments = []
        for m_spec, m_idx in zip(model_specs, model_indices, strict=True):
            base_name = self.config.name
            if multiple_models:
                short_m = get_model_short_name(m_spec)
                base_name = f"{base_name}-{short_m}"
            if multiple_priorities:
                base_name = f"{base_name}-{priority}"

            model_overrides = [self._build_model_overrides(m_spec, m_idx)]

            experiments.append(
                ExperimentPlan(
                    name=base_name,
                    model_specs=[m_spec],
                    priority=priority,
                    tasks=tasks,
                    original_task_specs=self.config.task_specs,
                    total_expanded_tasks=total_expanded_tasks,
                    num_gpus=self.config.gpus,
                    parallelism=1,
                    split_index=None,
                    total_splits=None,
                    model_overrides=model_overrides,
                    task_overrides=task_overrides,
                )
            )
        return experiments

    def build(self) -> tuple[list[ExperimentPlan], list[str]]:
        """Build the experiment plan."""
        from olmo_eval.core.configs import expand_tasks

        experiment_plan: list[ExperimentPlan] = []
        split_models: list[str] = []

        task_overrides = self.config.task_overrides
        model_specs = self.config.model_specs
        model_indices = list(range(len(model_specs)))

        multiple_models = len(model_specs) > 1
        multiple_priorities = len(self.tasks_by_priority) > 1

        for t_priority, t_list in self.tasks_by_priority.items():
            effective_priority = self.override_priority or t_priority
            total_expanded = len(expand_tasks(t_list))

            experiments = self._build_experiments(
                model_specs,
                model_indices,
                t_list,
                effective_priority,
                total_expanded,
                multiple_models,
                multiple_priorities,
                task_overrides,
            )
            experiment_plan.extend(experiments)

        return experiment_plan, split_models
