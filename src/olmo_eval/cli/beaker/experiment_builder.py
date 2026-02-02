"""Experiment plan building for Beaker launch."""

from __future__ import annotations

import json as json_module
from typing import TYPE_CHECKING

from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
from olmo_eval.core.types import RunnerType

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.cli.beaker.model_grouper import ModelGrouper
    from olmo_eval.launch import BeakerModelSpec


class ExperimentPlanBuilder:
    """Builds experiment plans with task splits."""

    def __init__(
        self,
        config: LaunchConfig,
        model_grouper: ModelGrouper,
        tasks_by_priority: dict[str, list[str]],
        agent_task_specs: set[str],
        override_priority: str | None = None,
    ):
        """Initialize the builder.

        Args:
            config: Parsed launch configuration.
            model_grouper: Model grouper instance.
            tasks_by_priority: Dict mapping priority -> task specs.
            agent_task_specs: Set of agent task specifications.
            override_priority: Priority extracted from -o priority=X, overrides default.
        """
        self.config = config
        self.model_grouper = model_grouper
        self.tasks_by_priority = tasks_by_priority
        self.agent_task_specs = agent_task_specs
        self.override_priority = override_priority

    def is_agent_spec(self, spec: str) -> bool:
        """Check if a task spec is an agent task."""
        from olmo_eval.core.configs import expand_tasks
        from olmo_eval.evals.tasks import get_base_task_name

        base_spec = get_base_task_name(spec)
        expanded = expand_tasks([base_spec])
        return all(t in self.agent_task_specs for t in expanded)

    def _get_model_gpu_needs(
        self, model_cfgs: list[BeakerModelSpec], model_specs: list[str]
    ) -> list[int]:
        """Get GPU needs (gpus * parallelism) for each model."""
        gpu_needs = []
        for m_cfg, m_spec in zip(model_cfgs, model_specs, strict=True):
            resources = self.model_grouper.get_model_resources(m_cfg, m_spec)
            gpu_needs.append(resources["gpus"] * resources["parallelism"])
        return gpu_needs

    def _build_model_overrides(
        self, m_cfg: BeakerModelSpec, m_spec: str, gpus: int, original_index: int
    ) -> list[str]:
        """Build list of -o override strings for a model."""
        overrides: list[str] = []

        # Include CLI overrides first (e.g., provider.name=vllm)
        # Use original_index to look up positional overrides
        cli_overrides: list[str] = []
        if original_index < len(self.config.model_overrides):
            cli_overrides = self.config.model_overrides[original_index]
            overrides.extend(cli_overrides)

        # Add GPU count only if not already specified in CLI overrides
        has_gpus_override = any(o.startswith("gpus=") for o in cli_overrides)
        if not has_gpus_override:
            overrides.append(f"gpus={gpus}")

        # Get model resources for additional overrides
        m_resources = self.model_grouper.get_model_resources(m_cfg, m_spec)

        if m_resources.get("load_format"):
            overrides.append(f"load_format={m_resources['load_format']}")

        if m_resources.get("extra_loader_config"):
            json_config = json_module.dumps(
                m_resources["extra_loader_config"], separators=(",", ":")
            )
            overrides.append(f"extra_loader_config={json_config}")

        return overrides

    def _build_experiments_no_pack(
        self,
        model_cfgs: list[BeakerModelSpec],
        model_specs: list[str],
        model_indices: list[int],
        tasks: list[str],
        priority: str,
        runner_type: RunnerType,
        total_expanded_tasks: int,
        multiple_models: bool,
        multiple_priorities: bool,
        task_overrides: dict[str, list[str]],
    ) -> list[ExperimentPlan]:
        """Build one experiment per model (no packing)."""
        from olmo_eval.launch import get_model_short_name

        experiments = []
        for m_cfg, m_spec, m_idx in zip(model_cfgs, model_specs, model_indices, strict=True):
            resources = self.model_grouper.get_model_resources(m_cfg, m_spec)
            m_gpus = resources["gpus"]
            m_parallelism = resources["parallelism"]
            total_model_gpus = m_gpus * m_parallelism

            # Build experiment name
            base_name = self.config.name
            if multiple_models:
                short_m = get_model_short_name(m_cfg)
                base_name = f"{base_name}-{short_m}"
            if multiple_priorities:
                base_name = f"{base_name}-{priority}"

            # Build model overrides
            model_overrides = [self._build_model_overrides(m_cfg, m_spec, m_gpus, m_idx)]

            experiments.append(
                ExperimentPlan(
                    name=base_name,
                    model_cfgs=[m_cfg],
                    model_specs=[m_spec],
                    priority=priority,
                    tasks=tasks,
                    original_task_specs=self.config.task_specs,
                    total_expanded_tasks=total_expanded_tasks,
                    model_gpu_counts=[m_gpus],
                    num_gpus=total_model_gpus,
                    parallelism=m_parallelism,
                    split_index=None,
                    total_splits=None,
                    runner_type=runner_type,
                    model_overrides=model_overrides,
                    task_overrides=task_overrides,
                )
            )
        return experiments

    def _build_experiments_packed(
        self,
        model_cfgs: list[BeakerModelSpec],
        model_specs: list[str],
        model_indices: list[int],
        tasks: list[str],
        priority: str,
        runner_type: RunnerType,
        total_expanded_tasks: int,
        multiple_models: bool,
        multiple_priorities: bool,
        task_overrides: dict[str, list[str]],
    ) -> tuple[list[ExperimentPlan], list[str]]:
        """Build experiments with models packed together when they fit."""
        from olmo_eval.launch import get_model_short_name

        experiments = []
        split_models: list[str] = []

        # Get GPU needs for each model
        gpu_needs = self._get_model_gpu_needs(model_cfgs, model_specs)
        total_gpus = sum(gpu_needs)

        # Build base experiment name
        base_name = self.config.name
        group_has_multiple_models = len(model_cfgs) > 1
        if not group_has_multiple_models and multiple_models:
            short_m = get_model_short_name(model_cfgs[0])
            base_name = f"{base_name}-{short_m}"
        if multiple_priorities:
            base_name = f"{base_name}-{priority}"

        # Get model GPU counts and overrides
        model_gpu_counts = [
            self.model_grouper.get_model_resources(cfg, spec)["gpus"]
            for cfg, spec in zip(model_cfgs, model_specs, strict=True)
        ]
        model_overrides = [
            self._build_model_overrides(cfg, spec, gpus, idx)
            for cfg, spec, gpus, idx in zip(
                model_cfgs, model_specs, model_gpu_counts, model_indices, strict=True
            )
        ]

        if total_gpus <= self.config.max_gpus_per_node:
            # All models fit together
            experiments.append(
                ExperimentPlan(
                    name=base_name,
                    model_cfgs=model_cfgs,
                    model_specs=model_specs,
                    priority=priority,
                    tasks=tasks,
                    original_task_specs=self.config.task_specs,
                    total_expanded_tasks=total_expanded_tasks,
                    model_gpu_counts=model_gpu_counts,
                    num_gpus=total_gpus,
                    parallelism=1,  # With packed models, parallelism is per-model
                    split_index=None,
                    total_splits=None,
                    runner_type=runner_type,
                    model_overrides=model_overrides,
                    task_overrides=task_overrides,
                )
            )
        else:
            # Need to split - use greedy bin packing
            # Each bin: (model_cfgs, model_specs, model_gpu_counts, model_overrides, total_gpus)
            bins: list[
                tuple[list[BeakerModelSpec], list[str], list[int], list[list[str]], int]
            ] = []

            for m_cfg, m_spec, gpu_need, m_gpus, m_overrides in zip(
                model_cfgs,
                model_specs,
                gpu_needs,
                model_gpu_counts,
                model_overrides,
                strict=True,
            ):
                placed = False
                for i, (bin_cfgs, bin_specs, bin_gpu_counts, bin_overrides, bin_gpus) in enumerate(
                    bins
                ):
                    if bin_gpus + gpu_need <= self.config.max_gpus_per_node:
                        bin_cfgs.append(m_cfg)
                        bin_specs.append(m_spec)
                        bin_gpu_counts.append(m_gpus)
                        bin_overrides.append(m_overrides)
                        bins[i] = (
                            bin_cfgs,
                            bin_specs,
                            bin_gpu_counts,
                            bin_overrides,
                            bin_gpus + gpu_need,
                        )
                        placed = True
                        break
                if not placed:
                    bins.append(([m_cfg], [m_spec], [m_gpus], [m_overrides], gpu_need))
                    split_models.append(m_cfg.name_or_path)

            total_splits = len(bins)
            for i, (bin_cfgs, bin_specs, bin_gpu_counts, bin_overrides, bin_gpus) in enumerate(
                bins
            ):
                exp_name = f"{base_name}-{i + 1:03d}" if total_splits > 1 else base_name
                experiments.append(
                    ExperimentPlan(
                        name=exp_name,
                        model_cfgs=bin_cfgs,
                        model_specs=bin_specs,
                        priority=priority,
                        tasks=tasks,
                        original_task_specs=self.config.task_specs,
                        total_expanded_tasks=total_expanded_tasks,
                        model_gpu_counts=bin_gpu_counts,
                        num_gpus=bin_gpus,
                        parallelism=1,
                        split_index=i + 1 if total_splits > 1 else None,
                        total_splits=total_splits if total_splits > 1 else None,
                        runner_type=runner_type,
                        model_overrides=bin_overrides,
                        task_overrides=task_overrides,
                    )
                )

        return experiments, split_models

    def build(self) -> tuple[list[ExperimentPlan], list[str]]:
        """Build the experiment plan.

        Returns:
            Tuple of (experiment_plan, split_models).
            - experiment_plan: List of ExperimentPlan objects.
            - split_models: List of model names that were split across experiments.
        """
        from olmo_eval.core.configs import expand_tasks

        experiment_plan: list[ExperimentPlan] = []
        split_models: list[str] = []

        # task_overrides is already filtered (priority extracted in launch.py)
        task_overrides = self.config.task_overrides

        model_groups = self.model_grouper.group()
        multiple_models = len(self.config.model_configs) > 1
        multiple_priorities = len(self.tasks_by_priority) > 1

        for model_group in model_groups:
            # model_group contains (config, spec, original_index) tuples
            group_model_cfgs = [cfg for cfg, _, _ in model_group]
            group_model_specs = [spec for _, spec, _ in model_group]
            group_model_indices = [idx for _, _, idx in model_group]

            for t_priority, t_list in self.tasks_by_priority.items():
                # Use override priority if specified via -o priority=X, else use task's priority
                effective_priority = self.override_priority or t_priority

                # Split tasks into agent and simple categories
                agent_tasks = [t for t in t_list if self.is_agent_spec(t)]
                simple_tasks = [t for t in t_list if not self.is_agent_spec(t)]

                for task_category, category_tasks in [
                    ("agent", agent_tasks),
                    ("simple", simple_tasks),
                ]:
                    if not category_tasks:
                        continue

                    category_expanded = len(expand_tasks(category_tasks))
                    # Agent tasks always use AGENT runner, others use config's runner_type
                    effective_runner = (
                        RunnerType.AGENT if task_category == "agent" else self.config.runner_type
                    )

                    if self.config.pack_models:
                        # Pack models together when they fit
                        experiments, splits = self._build_experiments_packed(
                            group_model_cfgs,
                            group_model_specs,
                            group_model_indices,
                            category_tasks,
                            effective_priority,
                            effective_runner,
                            category_expanded,
                            multiple_models,
                            multiple_priorities,
                            task_overrides,
                        )
                        experiment_plan.extend(experiments)
                        split_models.extend(splits)
                    else:
                        # Each model gets its own experiment (default)
                        experiments = self._build_experiments_no_pack(
                            group_model_cfgs,
                            group_model_specs,
                            group_model_indices,
                            category_tasks,
                            effective_priority,
                            effective_runner,
                            category_expanded,
                            multiple_models,
                            multiple_priorities,
                            task_overrides,
                        )
                        experiment_plan.extend(experiments)

        return experiment_plan, split_models
