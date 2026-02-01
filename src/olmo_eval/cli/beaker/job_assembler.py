"""Job configuration assembly for Beaker launch."""

from __future__ import annotations

import json as json_module
from typing import TYPE_CHECKING, Any

from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.launch import BeakerJobConfig, EvalConfig, ModelConfig


class JobConfigAssembler:
    """Assembles BeakerJobConfig from experiment plan."""

    def __init__(
        self,
        config: LaunchConfig,
        eval_config: EvalConfig | None,
        effective_image: str,
        effective_groups: list[str],
        beaker_username: str,
        common_secrets: list[tuple[str, str]],
        store_secrets: list[tuple[str, str]],
        task_secrets: list[tuple[str, str]],
        inject_aws_credentials: bool,
        inject_gcs_credentials: bool,
    ):
        """Initialize the assembler.

        Args:
            config: Parsed launch configuration.
            eval_config: Optional EvalConfig from YAML file.
            effective_image: Beaker image to use.
            effective_groups: List of Beaker groups.
            beaker_username: Beaker username.
            common_secrets: List of (env_var, secret_name) for common secrets.
            store_secrets: List of (env_var, secret_name) for storage secrets.
            task_secrets: List of (env_var, secret_name) for task secrets.
            inject_aws_credentials: Whether to inject AWS credentials.
            inject_gcs_credentials: Whether to inject GCS credentials.
        """
        self.config = config
        self.eval_config = eval_config
        self.effective_image = effective_image
        self.effective_groups = effective_groups
        self.beaker_username = beaker_username
        self.common_secrets = common_secrets
        self.store_secrets = store_secrets
        self.task_secrets = task_secrets
        self.inject_aws_credentials = inject_aws_credentials
        self.inject_gcs_credentials = inject_gcs_credentials

    def assemble(self, exp: dict[str, Any]) -> BeakerJobConfig:
        """Assemble a BeakerJobConfig for an experiment.

        Args:
            exp: Experiment plan dictionary.

        Returns:
            Configured BeakerJobConfig.
        """
        from olmo_eval.core.constants.infrastructure import (
            BACKEND_OPTIONAL_GROUPS,
        )
        from olmo_eval.launch import BeakerEnvSecret, BeakerJobConfig

        exp_model_cfgs: list[ModelConfig] = exp["model_cfgs"]
        exp_model_specs: list[str] = exp["model_specs"]
        exp_name: str = exp["name"]
        task_list: list[str] = exp["tasks"]
        exp_num_gpus: int = exp["num_gpus"]
        exp_parallelism: int = exp["parallelism"]
        effective_priority: str = exp["priority"]
        exp_is_agent: bool = exp.get("is_agent", False)

        first_model_cfg = exp_model_cfgs[0]

        # Get effective resources
        model_resources = self._get_model_resources(first_model_cfg)

        # Apply CLI overrides
        effective_cluster = self.config.cluster or str(model_resources.get("cluster", ""))
        effective_preemptible = (
            self.config.preemptible
            if self.config.preemptible is not None
            else bool(model_resources.get("preemptible", True))
        )
        effective_timeout = self.config.timeout or str(model_resources.get("timeout", "24h"))
        effective_shared_memory = str(model_resources.get("shared_memory") or "10GiB")

        # Build command
        command = self._build_command(
            exp_model_cfgs,
            exp_model_specs,
            task_list,
            exp_name,
            exp_parallelism,
            exp_num_gpus,
            exp_is_agent,
            model_resources,
        )

        # Determine provider extras
        if exp_is_agent:
            provider_extras = ["vllm", "agents"]
        else:
            config_provider = model_resources.get("provider") or "vllm"
            provider_group = BACKEND_OPTIONAL_GROUPS.get(str(config_provider))
            provider_extras = [provider_group] if provider_group else []

        install_extras = list(provider_extras)
        if self.config.store:
            install_extras.append("postgres")

        # Build secrets
        env_secrets = [
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in self.common_secrets
        ]
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in self.store_secrets
        )
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in self.task_secrets
        )

        # Build env vars
        job_env_vars = {
            "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
            "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
            "BEAKER_AUTHOR": self.beaker_username,
            "UV_LINK_MODE": "copy",
        }
        if self.config.uv_cache_dir:
            job_env_vars["UV_CACHE_DIR"] = self.config.uv_cache_dir

        return BeakerJobConfig(
            name=exp_name,
            command=command,
            cluster=effective_cluster,
            num_gpus=exp_num_gpus,
            priority=effective_priority,
            preemptible=effective_preemptible,
            timeout=effective_timeout,
            shared_memory=effective_shared_memory,
            retries=self.config.retries,
            workspace=self.config.workspace,
            budget=self.config.budget,
            extras=install_extras,
            groups=self.effective_groups,
            beaker_image=self.effective_image,
            inject_aws_credentials=self.inject_aws_credentials,
            inject_gcs_credentials=self.inject_gcs_credentials,
            env_vars=job_env_vars,
            env_secrets=env_secrets,
        )

    def _get_model_resources(self, m_cfg: ModelConfig) -> dict[str, Any]:
        """Get model resources from config or defaults."""
        if self.eval_config is not None:
            return self.eval_config.get_model_resources(m_cfg)
        return {
            "gpus": m_cfg.gpus or self.config.gpus,
            "parallelism": m_cfg.parallelism or self.config.parallelism,
            "cluster": m_cfg.cluster or self.config.cluster,
            "preemptible": m_cfg.preemptible if m_cfg.preemptible is not None else True,
            "timeout": m_cfg.timeout or self.config.timeout,
            "shared_memory": m_cfg.shared_memory,
            "provider": m_cfg.provider,
        }

    def _build_command(
        self,
        exp_model_cfgs: list[ModelConfig],
        exp_model_specs: list[str],
        task_list: list[str],
        exp_name: str,
        exp_parallelism: int,
        exp_num_gpus: int,
        exp_is_agent: bool,
        model_resources: dict[str, Any],
    ) -> list[str]:
        """Build the olmo-eval run command."""
        command: list[str] = ["olmo-eval", "run"]

        # Set output directory for Beaker (defaults to /tmp/results locally)
        command.extend(["-o", BEAKER_RESULT_DIR])

        # Add models
        for m_cfg, m_spec in zip(exp_model_cfgs, exp_model_specs, strict=True):
            m_resources = (
                self.eval_config.get_model_resources(m_cfg)
                if self.eval_config is not None
                else model_resources
            )

            final_model_spec = m_spec
            config_inline_overrides: list[str] = []

            if m_resources.get("load_format"):
                config_inline_overrides.append(f"load_format={m_resources['load_format']}")

            if m_resources.get("extra_loader_config"):
                json_config = json_module.dumps(
                    m_resources["extra_loader_config"], separators=(",", ":")
                )
                config_inline_overrides.append(f"extra_loader_config={json_config}")

            if config_inline_overrides:
                if "::" in final_model_spec:
                    final_model_spec = f"{final_model_spec},{','.join(config_inline_overrides)}"
                else:
                    final_model_spec = f"{final_model_spec}::{','.join(config_inline_overrides)}"

            command.extend(["-m", final_model_spec])

            if m_cfg.alias:
                command.extend(["--alias", m_cfg.alias])

        # Add tasks
        for t in task_list:
            command.extend(["-t", t])

        # Add parallelism
        if exp_parallelism > 1:
            command.extend(["--parallelism", str(exp_parallelism)])

        # Agent-specific or async flags
        if exp_is_agent:
            command.append("--agent")
            if exp_num_gpus > 1:
                command.extend(["--num-gpus", str(exp_num_gpus)])
        else:
            self._add_async_flags(command, model_resources)

        # Add S3 options
        if self.config.s3_bucket and self.config.s3_prefix:
            command.extend(["--s3-bucket", self.config.s3_bucket])
            command.extend(["--s3-prefix", self.config.s3_prefix])
            if self.effective_groups:
                command.extend(["--s3-group", self.effective_groups[0]])
            if self.config.s3_endpoint_url:
                command.extend(["--s3-endpoint-url", self.config.s3_endpoint_url])
            if self.config.s3_region != "us-east-1":
                command.extend(["--s3-region", self.config.s3_region])

        # Add experiment group
        if self.effective_groups:
            command.extend(["--experiment-group", self.effective_groups[0]])

        command.extend(["--experiment-name", exp_name])

        if self.config.store:
            command.append("--store")

        if self.config.debug_requests:
            command.append("--debug-requests")
        if self.config.debug_provider:
            command.append("--debug-provider")
        if not self.config.save_predictions:
            command.append("--no-save-predictions")
        if not self.config.save_requests:
            command.append("--no-save-requests")
        if self.config.inspect_instance:
            command.append("--inspect-instance")
        if self.config.inspect_formatted:
            command.append("--inspect-formatted")
        if self.config.inspect_tokens:
            command.append("--inspect-tokens")
        if self.config.inspect_response:
            command.append("--inspect-response")
        if self.config.inspect_request:
            command.append("--inspect-request")

        return command

    def _add_async_flags(self, command: list[str], model_resources: dict[str, Any]) -> None:
        """Add async-related flags to command."""
        effective_use_async = self.config.use_async or model_resources.get("use_async", False)
        effective_use_async_stream = self.config.use_async_stream or model_resources.get(
            "use_async_stream", False
        )
        effective_num_workers = (
            self.config.num_workers
            if self.config.num_workers is not None
            else model_resources.get("num_workers")
        )
        effective_gpus_per_worker = (
            self.config.gpus_per_worker
            if self.config.gpus_per_worker != 1
            else model_resources.get("gpus_per_worker", 1)
        )

        if effective_use_async_stream:
            command.append("--async-stream")
            if effective_num_workers is not None:
                command.extend(["--num-workers", str(effective_num_workers)])
            if effective_gpus_per_worker and effective_gpus_per_worker != 1:
                command.extend(["--gpus-per-worker", str(effective_gpus_per_worker)])
        elif effective_use_async:
            command.append("--async")
            if effective_num_workers is not None:
                command.extend(["--num-workers", str(effective_num_workers)])
            if effective_gpus_per_worker and effective_gpus_per_worker != 1:
                command.extend(["--gpus-per-worker", str(effective_gpus_per_worker)])
