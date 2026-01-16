"""Beaker launch utilities for olmo-eval.

This module provides a simplified API for launching evaluation jobs on Beaker.

Example:
    from olmo_eval.launch import BeakerJobConfig, BeakerLauncher

    config = BeakerJobConfig(
        name="eval-llama3-mmlu",
        command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
        cluster="h100",
        num_gpus=1,
    )

    launcher = BeakerLauncher()
    experiment = launcher.launch(config)
"""

from olmo_eval.launch.beaker import (
    BeakerEnvSecret,
    BeakerJobConfig,
    BeakerLauncher,
    BeakerWekaBucket,
    parse_task_with_priority,
    print_experiment_config,
    resolve_clusters,
    validate_priority_configuration,
)
from olmo_eval.launch.config import (
    LaunchConfig,
    ModelConfig,
    get_template,
    parse_model_config,
)

__all__ = [
    "BeakerEnvSecret",
    "BeakerJobConfig",
    "BeakerLauncher",
    "BeakerWekaBucket",
    "LaunchConfig",
    "ModelConfig",
    "get_template",
    "parse_model_config",
    "parse_task_with_priority",
    "print_experiment_config",
    "resolve_clusters",
    "validate_priority_configuration",
]
