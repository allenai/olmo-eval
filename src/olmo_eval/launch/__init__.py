"""Beaker launch utilities for olmo-eval.

This module provides a simplified API for launching evaluation jobs on Beaker.

Example:
    from olmo_eval.launch import BeakerJobConfig, BeakerLauncher

    config = BeakerJobConfig(
        name="eval-llama3-mmlu",
        command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
        cluster="ai2/ceres",
        workspace="ai2/oe-data",
        budget="ai2/oe-base",
        num_gpus=1,
    )

    launcher = BeakerLauncher(workspace="ai2/oe-data")
    experiment = launcher.launch(config)
"""

from olmo_eval.launch.beaker import (
    BeakerEnvSecret,
    BeakerJobConfig,
    BeakerLauncher,
    BeakerWekaBucket,
    calculate_experiment_splits,
    parse_task_with_priority,
    print_experiment_config,
    resolve_clusters,
    validate_priority_configuration,
)
from olmo_eval.launch.config import (
    EvalConfig,
    ModelConfig,
    get_model_short_name,
    get_tasks_short_name,
    get_template,
    parse_model_config,
)
from olmo_eval.launch.secrets import ensure_common_secrets

__all__ = [
    "ensure_common_secrets",
    "BeakerEnvSecret",
    "BeakerJobConfig",
    "BeakerLauncher",
    "BeakerWekaBucket",
    "EvalConfig",
    "ModelConfig",
    "calculate_experiment_splits",
    "get_model_short_name",
    "get_tasks_short_name",
    "get_template",
    "parse_model_config",
    "parse_task_with_priority",
    "print_experiment_config",
    "resolve_clusters",
    "validate_priority_configuration",
]
