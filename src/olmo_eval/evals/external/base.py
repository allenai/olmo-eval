"""Base class for external evaluations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from olmo_eval.evals.external.config import ExternalEvalConfig
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.inference.providers.config import ProviderConfig

logger = logging.getLogger(__name__)


class SandboxExecutor(Protocol):
    """Protocol for sandbox executor used in result extraction."""

    async def execute(self, command: str, timeout: float = 30.0) -> str:
        """Execute a command in the sandbox."""
        ...


class ExternalEval(ABC):
    """Abstract base class for external evaluations.

    External evaluations run inside sandbox containers and communicate
    with a model provider via an OpenAI-compatible API.

    Subclasses must implement `extract_results` to define how results
    are collected from the evaluation output.
    """

    def __init__(self, config: ExternalEvalConfig) -> None:
        """Initialize the external evaluation.

        Args:
            config: Configuration for the evaluation.
        """
        self.config = config

    @property
    def name(self) -> str:
        """Get the evaluation name."""
        return self.config.name

    @abstractmethod
    async def extract_results(
        self,
        executor: SandboxExecutor,
        run_output: str,
        exit_code: int,
    ) -> ExternalEvalResult:
        """Extract results from the evaluation execution.

        This method is called after the run command completes. Implementations
        should parse the output and/or read result files to build the result.

        Args:
            executor: Sandbox executor for reading files or running commands.
            run_output: Stdout/stderr from the run command.
            exit_code: Exit code from the run command (0 = success).

        Returns:
            Evaluation result with metrics, metadata, and success status.
        """
        ...

    @abstractmethod
    async def execute(
        self,
        provider_url: str,
        model_name: str,
        output_dir: str | None = None,
        container_runtime: str = "podman",
        use_network_host: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> ExternalEvalResult:
        """Execute the external evaluation.

        Args:
            provider_url: URL of the inference provider (already translated
                for container access if needed).
            model_name: Name/identifier of the model being evaluated.
            output_dir: Optional directory to write results.
            container_runtime: Container runtime to use (docker or podman).
            use_network_host: Whether to use --network=host for the container.
            extra_env: Additional environment variables to set.

        Returns:
            Result of the evaluation.
        """
        ...

    async def execute_with_provider(
        self,
        provider_config: ProviderConfig,
        output_dir: str | None = None,
        container_runtime: str = "podman",
        in_beaker: bool | None = None,
    ) -> ExternalEvalResult:
        """Execute the evaluation using a provider configuration.

        This is a convenience method that handles URL translation and
        extracts the model name from the provider config.

        Args:
            provider_config: Configuration for the inference provider.
            output_dir: Optional directory to write results.
            container_runtime: Container runtime to use.
            in_beaker: Whether running in Beaker. Auto-detected if None.

        Returns:
            Result of the evaluation.
        """
        from olmo_eval.evals.external.network import (
            get_provider_url_for_container,
            is_running_in_beaker,
        )

        if in_beaker is None:
            in_beaker = is_running_in_beaker()

        # Get the provider's base URL
        base_url = provider_config.base_url
        if base_url is None:
            # Default vLLM server port
            base_url = "http://localhost:8000"

        # Translate URL for container access
        provider_url = get_provider_url_for_container(
            base_url,
            runtime=container_runtime,
            in_beaker=in_beaker,
        )

        return await self.execute(
            provider_url=provider_url,
            model_name=provider_config.model,
            output_dir=output_dir,
            container_runtime=container_runtime,
            use_network_host=in_beaker,
        )
