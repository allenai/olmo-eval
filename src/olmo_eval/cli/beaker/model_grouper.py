"""Model grouping by runtime signature for Beaker launch."""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.launch import BeakerModelSpec, EvalConfig


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

    def get_model_resources(self, m_cfg: BeakerModelSpec, m_spec: str) -> dict[str, Any]:
        """Get runtime-critical resources for a model.

        Args:
            m_cfg: Model configuration.
            m_spec: Model specification string.

        Returns:
            Dict with gpus, parallelism, cluster, provider.
        """
        if self.eval_config is not None:
            m_resources = self.eval_config.get_model_resources(m_cfg)
            m_gpus = m_resources.get("gpus", 1)
            m_parallelism = m_resources.get("parallelism", 1)
            m_cluster = self.config.cluster or m_resources.get("cluster")
            m_provider = m_resources.get("provider")
        else:
            m_gpus = m_cfg.gpus
            m_parallelism = m_cfg.parallelism
            m_cluster = m_cfg.cluster or self.config.cluster
            m_provider = m_cfg.provider
        return {
            "gpus": m_gpus,
            "parallelism": m_parallelism,
            "cluster": m_cluster,
            "provider": m_provider,
        }

    def get_runtime_signature(self, m_cfg: BeakerModelSpec, m_spec: str) -> tuple:
        """Get a hashable runtime signature for grouping compatible models.

        Models with different GPU counts can be grouped together since they
        can run on the same node as long as total GPUs fit. The signature
        only includes settings that require incompatible runtime environments.

        Args:
            m_cfg: Model configuration.
            m_spec: Model specification string (unused, overrides come from BeakerModelSpec).

        Returns:
            Tuple that can be used as a grouping key.
        """
        resources = self.get_model_resources(m_cfg, m_spec)

        # Convert provider to string for comparison (ProviderConfig is not sortable)
        provider = resources["provider"]
        provider_str = provider.name if provider and hasattr(provider, "name") else str(provider)

        # gpus and parallelism are NOT part of signature - models with different
        # GPU counts can run together if they fit on the same node
        return (
            resources["cluster"],
            provider_str,
        )

    def group(self) -> list[list[tuple[BeakerModelSpec, str, int]]]:
        """Group models by runtime signature.

        Returns:
            List of model groups, where each group is a list of
            (config, spec, original_index) tuples. The original_index is the
            position in the original CLI model list, used for override lookup.
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
            # (config, spec, original_index) tuples
            groups.append([(g[2], g[3], g[1]) for g in group_list])

        return groups
