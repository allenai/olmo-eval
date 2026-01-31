"""Model grouping by runtime signature for Beaker launch."""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.launch import EvalConfig, ModelConfig


class ModelGrouper:
    """Groups models by compatible runtime signature."""

    def __init__(
        self,
        config: LaunchConfig,
        eval_config: EvalConfig | None,
    ):
        """Initialize the grouper.

        Args:
            config: Parsed launch configuration.
            eval_config: Optional EvalConfig from YAML file.
        """
        self.config = config
        self.eval_config = eval_config

    def get_model_resources(self, m_cfg: ModelConfig, m_spec: str) -> dict[str, Any]:
        """Get runtime-critical resources for a model.

        Args:
            m_cfg: Model configuration.
            m_spec: Model specification string.

        Returns:
            Dict with gpus, parallelism, cluster, provider.
        """
        if self.eval_config is not None:
            m_resources = self.eval_config.get_model_resources(m_cfg)
            m_gpus = self.config.gpus if self.config.gpus != 1 else m_resources.get("gpus", 1)
            m_parallelism = (
                self.config.parallelism
                if self.config.parallelism != 1
                else m_resources.get("parallelism", 1)
            )
            m_cluster = self.config.cluster or m_resources.get("cluster")
            m_provider = m_resources.get("provider")
        else:
            m_gpus = m_cfg.gpus or self.config.gpus
            m_parallelism = m_cfg.parallelism or self.config.parallelism
            m_cluster = m_cfg.cluster or self.config.cluster
            m_provider = m_cfg.provider
        return {
            "gpus": m_gpus,
            "parallelism": m_parallelism,
            "cluster": m_cluster,
            "provider": m_provider,
        }

    def get_runtime_signature(self, m_cfg: ModelConfig, m_spec: str) -> tuple:
        """Get a hashable runtime signature for grouping compatible models.

        Args:
            m_cfg: Model configuration.
            m_spec: Model specification string.

        Returns:
            Tuple that can be used as a grouping key.
        """
        resources = self.get_model_resources(m_cfg, m_spec)

        # Extract inline overrides from spec
        _, _, inline_overrides = m_spec.partition("::")

        return (
            resources["gpus"],
            resources["parallelism"],
            resources["cluster"],
            resources["provider"],
            inline_overrides,
        )

    def group(self) -> list[list[tuple[ModelConfig, str]]]:
        """Group models by runtime signature.

        Returns:
            List of model groups, where each group is a list of (config, spec) tuples.
        """
        model_configs = self.config.model_configs
        model_specs = self.config.model_specs

        # Build (signature, index, config, spec) tuples
        model_data = [
            (self.get_runtime_signature(m_cfg, m_spec), i, m_cfg, m_spec)
            for i, (m_cfg, m_spec) in enumerate(zip(model_configs, model_specs, strict=True))
        ]

        # Sort by signature, then by original index
        sorted_models = sorted(model_data, key=lambda x: (x[0], x[1]))

        # Group by signature
        groups = []
        for _, group_iter in groupby(sorted_models, key=lambda x: x[0]):
            group_list = list(group_iter)
            groups.append([(g[2], g[3]) for g in group_list])  # (config, spec) pairs

        return groups
