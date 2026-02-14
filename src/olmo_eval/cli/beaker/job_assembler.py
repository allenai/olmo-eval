"""Job configuration assembly for Beaker launch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmo_eval.common.constants.infrastructure import (
    BEAKER_DEFAULT_BUDGET,
    BEAKER_RESULT_DIR,
    cluster_has_weka,
)
from olmo_eval.launch.beaker.mirror import log

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.config_loader import LaunchConfig
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
    from olmo_eval.launch import BeakerJobConfig


def get_provider_extras(model_spec: str, default_kind: str | None = None) -> list[str]:
    """Get the pip extras required for a model's provider.

    Args:
        model_spec: Model name or path.
        default_kind: Default provider kind if model is not a preset.

    Returns:
        List of pip extras needed for the provider.
    """
    from olmo_eval.common.configs import get_provider_config
    from olmo_eval.common.constants.infrastructure import BACKEND_OPTIONAL_GROUPS

    try:
        provider_config = get_provider_config(model_spec)
        provider_kind = provider_config.kind
    except Exception:
        provider_kind = default_kind

    if provider_kind:
        provider_extra = BACKEND_OPTIONAL_GROUPS.get(provider_kind)
        if provider_extra:
            return [provider_extra]
    return []


def assemble_external_eval_job(
    name: str,
    model: str,
    external_evals: list[str],
    cluster: str,
    num_gpus: int,
    workspace: str,
    beaker_image: str,
    priority: str = "normal",
    timeout: str = "24h",
    budget: str | None = None,
    groups: list[str] | None = None,
    tensor_parallel_size: int = 1,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    s3_region: str = "us-east-1",
    env_secrets: list[tuple[str, str]] | None = None,
    inject_aws_credentials: bool = False,
    inject_gcs_credentials: bool = False,
    eval_args: dict[str, str] | None = None,
) -> Any:
    """Assemble a BeakerJobConfig for running external evaluations.

    Args:
        name: Experiment name.
        model: Model name or path.
        external_evals: List of external evaluation names.
        cluster: Beaker cluster name.
        num_gpus: Number of GPUs.
        workspace: Beaker workspace.
        beaker_image: Container image to use.
        priority: Job priority.
        timeout: Job timeout.
        budget: Beaker budget.
        groups: Beaker groups.
        tensor_parallel_size: Tensor parallel size for vLLM.
        s3_bucket: S3 bucket for results.
        s3_prefix: S3 prefix for results.
        s3_region: S3 region.
        env_secrets: List of (env_var, secret_name) tuples.
        inject_aws_credentials: Whether to inject AWS credentials.
        inject_gcs_credentials: Whether to inject GCS credentials.
        eval_args: Arguments to pass to external evaluations.

    Returns:
        Configured BeakerJobConfig.
    """
    from olmo_eval.launch import BeakerEnvSecret, BeakerJobConfig

    # Build command
    command: list[str] = ["olmo-eval", "run-external"]
    command.extend(["-m", model])
    for eval_name in external_evals:
        command.extend(["-e", eval_name])
    command.extend(["-O", BEAKER_RESULT_DIR])

    if tensor_parallel_size > 1:
        command.extend(["--tp", str(tensor_parallel_size)])

    # Add eval_args
    if eval_args:
        for key, value in eval_args.items():
            command.extend(["-a", f"{key}={value}"])

    # Environment variables
    env_vars: dict[str, str] = {
        "BEAKER_ALLOW_SUBCONTAINERS": "1",
        "BEAKER_SKIP_DOCKER_SOCKET": "1",
    }

    if cluster_has_weka(cluster):
        env_vars.update(
            {
                "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
                "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
                "UV_LINK_MODE": "copy",
            }
        )

    # Get registry mirror URL
    try:
        from olmo_eval.launch.beaker.mirror import get_registry_mirror_url

        mirror_url = get_registry_mirror_url()
        env_vars["MIRROR_HOSTS"] = mirror_url
        setup_registry_mirror = True
    except Exception:
        setup_registry_mirror = False

    # Build env secrets
    beaker_env_secrets = []
    if env_secrets:
        beaker_env_secrets = [
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in env_secrets
        ]

    extras = ["sandbox"]
    provider_extras = get_provider_extras(model, default_kind="vllm_server")
    for extra in provider_extras:
        if extra not in extras:
            extras.append(extra)

    return BeakerJobConfig(
        name=name,
        command=command,
        cluster=cluster,
        num_gpus=num_gpus,
        priority=priority,
        timeout=timeout,
        shared_memory="10GiB",
        workspace=workspace,
        budget=budget or BEAKER_DEFAULT_BUDGET,
        groups=groups or [],
        beaker_image=beaker_image,
        inject_aws_credentials=inject_aws_credentials,
        inject_gcs_credentials=inject_gcs_credentials,
        env_vars=env_vars,
        env_secrets=beaker_env_secrets,
        enable_sandbox=True,
        setup_registry_mirror=setup_registry_mirror,
        extras=extras,
    )


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
        enable_sandbox: bool = False,
        secret_env_overrides: dict[str, str] | None = None,
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
        self.enable_sandbox = enable_sandbox
        self.secret_env_overrides = secret_env_overrides or {}

    def assemble(self, exp: ExperimentPlan) -> BeakerJobConfig:
        """Assemble a BeakerJobConfig for an experiment."""
        from olmo_eval.launch import BeakerEnvSecret, BeakerJobConfig

        command = self._build_command(exp)

        install_extras: list[str] = []
        if self.config.store:
            install_extras.append("postgres")

        if self.config.harness:
            from olmo_eval.harness import get_backend_extras, get_harness_preset

            preset = get_harness_preset(self.config.harness)
            if preset.backend:
                backend_extras = get_backend_extras(preset.backend)
                install_extras.extend(backend_extras)
            # Install sandbox extra for any sandbox mode (local, docker, or modal)
            if preset.sandboxes:
                install_extras.append("sandbox")

        # Get provider extras from model preset (takes precedence over harness default)
        for extra in get_provider_extras(exp.model_spec):
            if extra not in install_extras:
                install_extras.append(extra)

        # Collect env vars that have explicit overrides
        overridden_env_vars = set(self.secret_env_overrides.values())

        # Add default secrets, skipping any that are overridden
        env_secrets = [
            BeakerEnvSecret(env_var, secret_name)
            for env_var, secret_name in self.common_secrets
            if env_var not in overridden_env_vars
        ]
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name)
            for env_var, secret_name in self.store_secrets
            if env_var not in overridden_env_vars
        )
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name)
            for env_var, secret_name in self.task_secrets
            if env_var not in overridden_env_vars
        )
        # Add explicit secret overrides (beaker_secret -> env_var)
        env_secrets.extend(
            BeakerEnvSecret(env_var, beaker_secret)
            for beaker_secret, env_var in self.secret_env_overrides.items()
        )

        job_env_vars: dict[str, str] = {
            "BEAKER_AUTHOR": self.beaker_username,
            "BEAKER_WORKSPACE": self.config.workspace,
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

        # Configure sandbox environment and registry mirror
        setup_registry_mirror = False
        log.info(f"Sandbox enabled: {self.enable_sandbox}")
        if self.enable_sandbox:
            job_env_vars["BEAKER_ALLOW_SUBCONTAINERS"] = "1"
            job_env_vars["BEAKER_SKIP_DOCKER_SOCKET"] = "1"

            # Get registry mirror URL for faster image pulls (raises if unavailable)
            from olmo_eval.launch.beaker.mirror import get_registry_mirror_url

            mirror_url = get_registry_mirror_url()
            job_env_vars["MIRROR_HOSTS"] = mirror_url
            setup_registry_mirror = True

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
            enable_sandbox=self.enable_sandbox,
            setup_registry_mirror=setup_registry_mirror,
        )

    def _extract_task_dependencies(
        self, task_specs: list[str], task_overrides: dict[str, list[str]]
    ) -> list[str] | None:
        from olmo_eval.common.configs import expand_tasks
        from olmo_eval.evals.tasks.common import get_task_dependencies, parse_overrides

        # Expand suites to individual tasks before extracting dependencies
        expanded_specs = expand_tasks(task_specs)

        # Get dependencies from registered task configs
        deps = get_task_dependencies(expanded_specs)

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
        # Use --inspect if all inspect flags are enabled, otherwise add individual flags
        all_inspect = (
            self.config.inspect_instance
            and self.config.inspect_formatted
            and self.config.inspect_tokens
            and self.config.inspect_response
            and self.config.inspect_request
        )
        if all_inspect:
            command.append("--inspect")
        else:
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
