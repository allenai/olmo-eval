"""Job configuration assembly for Beaker launch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR, cluster_has_weka

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
    from olmo_eval.launch import BeakerJobConfig


class JobConfigAssembler:
    """Assembles BeakerJobConfig from experiment plan."""

    def __init__(
        self,
        config: LaunchConfig,
        effective_image: str,
        effective_groups: list[str],
        beaker_username: str,
        common_secrets: list[tuple[str, str]],
        store_secrets: list[tuple[str, str]],
        task_secrets: list[tuple[str, str]],
        inject_aws_credentials: bool,
        inject_gcs_credentials: bool,
    ):
        self.config = config
        self.effective_image = effective_image
        self.effective_groups = effective_groups
        self.beaker_username = beaker_username
        self.common_secrets = common_secrets
        self.store_secrets = store_secrets
        self.task_secrets = task_secrets
        self.inject_aws_credentials = inject_aws_credentials
        self.inject_gcs_credentials = inject_gcs_credentials

    def assemble(self, exp: ExperimentPlan) -> BeakerJobConfig:
        """Assemble a BeakerJobConfig for an experiment."""
        from olmo_eval.core.constants.infrastructure import BACKEND_OPTIONAL_GROUPS
        from olmo_eval.launch import BeakerEnvSecret, BeakerJobConfig

        command = self._build_command(exp)

        install_extras: list[str] = []
        if self.config.store:
            install_extras.append("postgres")

        if self.config.harness:
            from olmo_eval.core.harness import get_backend_extras, get_harness_preset

            preset = get_harness_preset(self.config.harness)
            backend_extras = get_backend_extras(preset.backend)
            install_extras.extend(backend_extras)
            provider_group = BACKEND_OPTIONAL_GROUPS.get(preset.provider.kind)
            if provider_group:
                install_extras.append(provider_group)

        env_secrets = [
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in self.common_secrets
        ]
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in self.store_secrets
        )
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in self.task_secrets
        )

        job_env_vars: dict[str, str] = {
            "BEAKER_AUTHOR": self.beaker_username,
        }

        if cluster_has_weka(self.config.cluster):
            job_env_vars.update(
                {
                    "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
                    "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
                    "UV_LINK_MODE": "copy",
                }
            )
            if self.config.uv_cache_dir:
                job_env_vars["UV_CACHE_DIR"] = self.config.uv_cache_dir

        task_packages = self._extract_task_dependencies(exp.tasks, exp.task_overrides)

        return BeakerJobConfig(
            name=exp.name,
            command=command,
            cluster=self.config.cluster,
            num_gpus=exp.num_gpus,
            priority=exp.priority,
            preemptible=self.config.preemptible,
            timeout=self.config.timeout,
            shared_memory="10GiB",
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
            task_packages=task_packages,
        )

    def _extract_task_dependencies(
        self, task_specs: list[str], task_overrides: dict[str, list[str]]
    ) -> list[str] | None:
        from olmo_eval.evals.tasks import get_task_dependencies
        from olmo_eval.evals.tasks.core.registry import parse_overrides

        deps = get_task_dependencies(task_specs)

        for _task_spec, overrides in task_overrides.items():
            for override_str in overrides:
                parsed = parse_overrides(override_str)
                if "dependencies" in parsed:
                    override_deps = parsed["dependencies"]
                    if isinstance(override_deps, list):
                        deps.extend(override_deps)
                    else:
                        deps.append(override_deps)

        deps = list(dict.fromkeys(deps))
        return deps if deps else None

    def _build_command(self, exp: ExperimentPlan) -> list[str]:
        """Build the olmo-eval run command."""
        command: list[str] = ["olmo-eval", "run"]

        command.extend(["-O", BEAKER_RESULT_DIR])

        command.extend(["-m", exp.model_spec])

        for t in exp.tasks:
            command.extend(["-t", t])
            if t in exp.task_overrides:
                for override in exp.task_overrides[t]:
                    command.extend(["-o", override])

        if exp.parallelism > 1:
            command.extend(["--parallelism", str(exp.parallelism)])

        self._add_worker_flags(command)

        if self.config.s3_bucket and self.config.s3_prefix:
            command.extend(["--s3-bucket", self.config.s3_bucket])
            command.extend(["--s3-prefix", self.config.s3_prefix])
            if self.effective_groups:
                command.extend(["--s3-group", self.effective_groups[0]])
            if self.config.s3_endpoint_url:
                command.extend(["--s3-endpoint-url", self.config.s3_endpoint_url])
            if self.config.s3_region != "us-east-1":
                command.extend(["--s3-region", self.config.s3_region])

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

        if self.config.harness:
            command.extend(["--harness", self.config.harness])

        for override in self.config.harness_overrides:
            command.extend(["-o", override])

        return command

    def _add_worker_flags(self, command: list[str]) -> None:
        if self.config.num_workers is not None:
            command.extend(["--num-workers", str(self.config.num_workers)])
        if self.config.gpus_per_worker != 1:
            command.extend(["--gpus-per-worker", str(self.config.gpus_per_worker)])
