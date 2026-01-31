"""Experiment plan building for Beaker launch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.cli.beaker.model_grouper import ModelGrouper


class ExperimentPlanBuilder:
    """Builds experiment plans with task splits."""

    def __init__(
        self,
        config: LaunchConfig,
        model_grouper: ModelGrouper,
        tasks_by_priority: dict[str, list[str]],
        agent_task_specs: set[str],
    ):
        """Initialize the builder.

        Args:
            config: Parsed launch configuration.
            model_grouper: Model grouper instance.
            tasks_by_priority: Dict mapping priority -> task specs.
            agent_task_specs: Set of agent task specifications.
        """
        self.config = config
        self.model_grouper = model_grouper
        self.tasks_by_priority = tasks_by_priority
        self.agent_task_specs = agent_task_specs

    def is_agent_spec(self, spec: str) -> bool:
        """Check if a task spec is an agent task."""
        from olmo_eval.core.configs import expand_tasks
        from olmo_eval.evals.tasks import get_base_task_name

        base_spec = get_base_task_name(spec)
        expanded = expand_tasks([base_spec])
        return all(t in self.agent_task_specs for t in expanded)

    def build(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Build the experiment plan.

        Returns:
            Tuple of (experiment_plan, split_models).
            - experiment_plan: List of experiment dicts.
            - split_models: List of model names that were split across experiments.
        """
        from olmo_eval.core.configs import expand_tasks
        from olmo_eval.launch import calculate_experiment_splits, get_model_short_name

        experiment_plan: list[dict[str, Any]] = []
        split_models: list[str] = []

        model_groups = self.model_grouper.group()
        multiple_models = len(self.config.model_configs) > 1
        multiple_priorities = len(self.tasks_by_priority) > 1

        for model_group in model_groups:
            group_model_cfgs = [cfg for cfg, _ in model_group]
            group_model_specs = [spec for _, spec in model_group]

            # Use first model's resources
            first_cfg = group_model_cfgs[0]
            first_spec = group_model_specs[0]
            resources = self.model_grouper.get_model_resources(first_cfg, first_spec)
            m_gpus = resources["gpus"]
            m_parallelism = resources["parallelism"]

            group_has_multiple_models = len(group_model_cfgs) > 1

            for t_priority, t_list in self.tasks_by_priority.items():
                # Split tasks into agent and simple categories
                agent_tasks = [t for t in t_list if self.is_agent_spec(t)]
                simple_tasks = [t for t in t_list if not self.is_agent_spec(t)]

                for task_category, category_tasks in [
                    ("agent", agent_tasks),
                    ("simple", simple_tasks),
                ]:
                    if not category_tasks:
                        continue

                    # Build experiment name
                    base_name = self.config.name
                    if not group_has_multiple_models and multiple_models:
                        short_m = get_model_short_name(first_cfg)
                        base_name = f"{base_name}-{short_m}"
                    if multiple_priorities:
                        base_name = f"{base_name}-{t_priority}"

                    splits = calculate_experiment_splits(
                        tasks=category_tasks,
                        gpus_per_model=m_gpus,
                        parallelism=m_parallelism,
                        max_gpus_per_node=self.config.max_gpus_per_node,
                    )

                    if len(splits) > 1:
                        for m_cfg in group_model_cfgs:
                            split_models.append(m_cfg.name_or_path)

                    category_expanded = len(expand_tasks(category_tasks))
                    num_models_in_group = len(group_model_cfgs)
                    total_splits = len(splits)

                    for i, split in enumerate(splits):
                        exp_name = f"{base_name}-{i + 1:03d}" if total_splits > 1 else base_name
                        total_gpus_for_group = split["num_gpus"] * num_models_in_group

                        experiment_plan.append(
                            {
                                "name": exp_name,
                                "model_cfgs": group_model_cfgs,
                                "model_specs": group_model_specs,
                                "priority": t_priority,
                                "tasks": split["tasks"],
                                "original_task_specs": self.config.task_specs,
                                "total_expanded_tasks": category_expanded,
                                "gpus_per_model": m_gpus,
                                "num_gpus": total_gpus_for_group,
                                "parallelism": split["parallelism"],
                                "split_index": i + 1 if total_splits > 1 else None,
                                "total_splits": total_splits if total_splits > 1 else None,
                                "is_agent": task_category == "agent",
                            }
                        )

        return experiment_plan, split_models
