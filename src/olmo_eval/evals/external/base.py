"""Base class for external evaluations."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.inference.providers.config import ProviderConfig

logger = logging.getLogger(__name__)


class ExternalEval(ABC):
    """Abstract base class for external evaluations.

    External evaluations are black-box benchmarks that run in sandbox
    containers and communicate with a model via an OpenAI-compatible API.

    Each evaluation implements its own execution logic, argument handling,
    and result extraction. There are no assumptions about setup commands,
    run commands, or result formats - those are implementation details.
    """

    # --- Abstract properties (must be implemented) ---

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this evaluation."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this evaluation measures."""
        ...

    @property
    @abstractmethod
    def sandbox_image(self) -> str:
        """Docker image to use for the sandbox container."""
        ...

    @property
    @abstractmethod
    def working_dir(self) -> str:
        """Working directory inside the sandbox container."""
        ...

    @property
    @abstractmethod
    def timeout_seconds(self) -> float:
        """Maximum execution time in seconds."""
        ...

    @property
    @abstractmethod
    def setup_commands(self) -> tuple[str, ...]:
        """Commands run to set up the evaluation environment."""
        ...

    @property
    @abstractmethod
    def run_command(self) -> str:
        """Command used to run the evaluation."""
        ...

    # --- Optional properties (can be overridden) ---

    @property
    def results_dir(self) -> str:
        """Directory where evaluation results are saved."""
        return f"{self.working_dir}/results"

    @property
    def required_secrets(self) -> tuple[str, ...]:
        """Environment variable names that must be forwarded to the sandbox."""
        return ()

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        """Arguments that can be passed to this evaluation.

        Returns a dict of arg_name -> (description, default_value).
        Use None for default_value if the argument is optional with no default.
        """
        return {}

    # --- Abstract methods ---

    @abstractmethod
    async def execute(
        self,
        provider_url: str,
        model_name: str,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
        use_host_network: bool = False,
        provider_kind: str | None = None,
    ) -> ExternalEvalResult:
        """Execute the external evaluation.

        Args:
            provider_url: URL of the inference provider (translated for
                container access).
            model_name: Name/identifier of the model being evaluated.
            args: Evaluation-specific arguments (e.g., domain, num_trials).
            output_dir: Optional directory to write results.
            container_runtime: Container runtime to use (docker or podman).
            use_host_network: Whether to use host networking for containers.
            provider_kind: Type of provider (e.g., "vllm_server", "litellm").
                Used to determine if local server setup is needed.

        Returns:
            Result of the evaluation.
        """
        ...

    # --- Convenience methods ---

    async def execute_with_provider(
        self,
        provider_config: ProviderConfig,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
        use_host_network: bool | None = None,
    ) -> ExternalEvalResult:
        """Execute the evaluation using a provider configuration."""
        from olmo_eval.evals.external.network import (
            get_provider_url_for_container,
            should_use_host_network,
        )

        if use_host_network is None:
            use_host_network = should_use_host_network()

        base_url = provider_config.base_url or "http://localhost:8000"
        provider_url = get_provider_url_for_container(
            base_url,
            runtime=container_runtime,
            use_host_network=use_host_network,
        )

        return await self.execute(
            provider_url=provider_url,
            model_name=provider_config.model,
            args=args,
            output_dir=output_dir,
            container_runtime=container_runtime,
            use_host_network=use_host_network,
            provider_kind=provider_config.kind,
        )

    # --- Helper methods (can be overridden) ---

    def _build_env_vars(self) -> dict[str, str]:
        """Build environment variables for the sandbox, validating required secrets."""
        env_vars: dict[str, str] = {}
        missing = []
        for secret in self.required_secrets:
            value = os.environ.get(secret)
            if value:
                env_vars[secret] = value
            else:
                missing.append(secret)

        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")

        return env_vars

    async def _run_setup(
        self, executor: Any, all_output: list[str], start_time: float
    ) -> ExternalEvalResult | None:
        """Run setup commands. Returns error result if any fail, None on success."""
        for cmd in self.setup_commands:
            logger.info(f"[{self.name}] Setup: {cmd}")
            result = await executor.execute_command(cmd, timeout=self.timeout_seconds)
            all_output.append(f"$ {cmd}\n{result.output}")

            if not result.success:
                return ExternalEvalResult(
                    name=self.name,
                    success=False,
                    error=f"Setup failed: {cmd}",
                    raw_output="\n".join(all_output),
                    duration_seconds=time.time() - start_time,
                )
        return None

    def _error_result(
        self, error: str, start_time: float, raw_output: str = ""
    ) -> ExternalEvalResult:
        """Create an error result."""
        return ExternalEvalResult(
            name=self.name,
            success=False,
            error=error,
            raw_output=raw_output,
            duration_seconds=time.time() - start_time,
        )

    def _create_sandbox_config(
        self,
        container_runtime: str,
        use_host_network: bool,
    ) -> Any:
        """Create sandbox configuration for this evaluation."""
        from olmo_eval.evals.external.network import get_docker_network_args
        from olmo_eval.harness.sandbox.config import (
            ContainerRuntime,
            SandboxConfig,
            SandboxMode,
        )

        runtime = cast(ContainerRuntime, container_runtime)
        return SandboxConfig(
            image=self.sandbox_image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            command_timeout=self.timeout_seconds,
            working_dir=self.working_dir,
            environment=tuple(self._build_env_vars().items()),
            docker_args=tuple(
                get_docker_network_args(
                    runtime=container_runtime, use_host_network=use_host_network
                )
            ),
        )

    def _save_results(self, result: ExternalEvalResult, output_dir: str) -> None:
        """Save results to the output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / f"{self.name}_results.json"

        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"[{self.name}] Results saved to {results_file}")
