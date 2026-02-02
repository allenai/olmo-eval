"""Job configuration assembly for Beaker launch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.core.types import RunnerType

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
    from olmo_eval.launch import BeakerJobConfig, BeakerModelSpec, EvalConfig


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

    def assemble(self, exp: ExperimentPlan) -> BeakerJobConfig:
        """Assemble a BeakerJobConfig for an experiment.

        Args:
            exp: Experiment plan object.

        Returns:
            Configured BeakerJobConfig.
        """
        from olmo_eval.core.constants.infrastructure import (
            BACKEND_OPTIONAL_GROUPS,
        )
        from olmo_eval.launch import BeakerEnvSecret, BeakerJobConfig

        first_model_cfg = exp.model_cfgs[0]

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
        command = self._build_command(exp, model_resources)

        # Determine provider extras
        if exp.runner_type == RunnerType.AGENT:
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

        # Extract task dependencies (from both registered configs and CLI overrides)
        task_packages = self._extract_task_dependencies(exp.tasks, exp.task_overrides)

        return BeakerJobConfig(
            name=exp.name,
            command=command,
            cluster=effective_cluster,
            num_gpus=exp.num_gpus,
            priority=exp.priority,
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
            provider_package=model_resources.get("provider_package"),
            task_packages=task_packages,
        )

    def _extract_task_dependencies(
        self, task_specs: list[str], task_overrides: dict[str, list[str]]
    ) -> list[str] | None:
        """Extract dependencies from task specs and CLI overrides.

        Args:
            task_specs: List of task specification strings.
            task_overrides: Dict mapping task specs to lists of override strings.

        Returns:
            List of package dependencies to install, or None if no dependencies.
        """
        from olmo_eval.evals.tasks import get_task_dependencies
        from olmo_eval.evals.tasks.core.registry import parse_overrides

        # Get dependencies from registered task configs
        deps = get_task_dependencies(task_specs)

        # Also extract dependencies from CLI overrides
        for _task_spec, overrides in task_overrides.items():
            for override_str in overrides:
                parsed = parse_overrides(override_str)
                if "dependencies" in parsed:
                    override_deps = parsed["dependencies"]
                    if isinstance(override_deps, list):
                        deps.extend(override_deps)
                    else:
                        deps.append(override_deps)

        # Deduplicate while preserving order
        deps = list(dict.fromkeys(deps))
        return deps if deps else None

    def _get_model_resources(self, m_cfg: BeakerModelSpec) -> dict:
        """Get model resources from config or defaults."""
        if self.eval_config is not None:
            return self.eval_config.get_model_resources(m_cfg)

        # Extract provider name and package from ProviderConfig
        provider_name = m_cfg.provider.name if m_cfg.provider else None
        provider_package = m_cfg.provider.package if m_cfg.provider else None

        # Model always has gpus and parallelism (default 1)
        return {
            "gpus": m_cfg.gpus,
            "parallelism": m_cfg.parallelism,
            "cluster": m_cfg.cluster or self.config.cluster,
            "preemptible": m_cfg.preemptible if m_cfg.preemptible is not None else True,
            "timeout": m_cfg.timeout or self.config.timeout,
            "shared_memory": m_cfg.shared_memory,
            "provider": provider_name,
            "provider_package": provider_package,
        }

    def _build_command(
        self,
        exp: ExperimentPlan,
        model_resources: dict,
    ) -> list[str]:
        """Build the olmo-eval run command."""
        command: list[str] = ["olmo-eval", "run"]

        # Set output directory for Beaker (use -O short form)
        command.extend(["-O", BEAKER_RESULT_DIR])

        # Add models with their overrides using -o flags
        for i, (m_cfg, m_spec) in enumerate(zip(exp.model_cfgs, exp.model_specs, strict=True)):
            # Add the model
            command.extend(["-m", m_spec])

            # Add per-model overrides using -o flags
            if i < len(exp.model_overrides):
                for override in exp.model_overrides[i]:
                    command.extend(["-o", override])

            # Add alias if present
            if m_cfg.alias:
                command.extend(["--alias", m_cfg.alias])

        # Add tasks with their overrides
        for t in exp.tasks:
            command.extend(["-t", t])
            # Add per-task overrides using -o flags
            if t in exp.task_overrides:
                for override in exp.task_overrides[t]:
                    command.extend(["-o", override])

        # Add parallelism
        if exp.parallelism > 1:
            command.extend(["--parallelism", str(exp.parallelism)])

        # Add runner type and related flags
        self._add_runner_flags(command, exp, model_resources)

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

        command.extend(["--experiment-name", exp.name])

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

    def _add_runner_flags(
        self, command: list[str], exp: ExperimentPlan, model_resources: dict
    ) -> None:
        """Add runner type and related flags to command."""

        runner_type = exp.runner_type

        # Add runner type flag (only if not sync, since sync is the default)
        if runner_type != RunnerType.SYNC:
            command.extend(["--runner-type", runner_type.value])

        # Agent-specific flags
        if runner_type == RunnerType.AGENT:
            if exp.num_gpus > 1:
                command.extend(["--num-gpus", str(exp.num_gpus)])
            return

        # Async/async-stream worker flags
        if runner_type in (RunnerType.ASYNC, RunnerType.ASYNC_STREAM):
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

            if effective_num_workers is not None:
                command.extend(["--num-workers", str(effective_num_workers)])
            if effective_gpus_per_worker and effective_gpus_per_worker != 1:
                command.extend(["--gpus-per-worker", str(effective_gpus_per_worker)])
