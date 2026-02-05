"""Runner factory for creating evaluation runners."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from olmo_eval.cli.run.config import RunConfig
from olmo_eval.core.types import RunnerType
from olmo_eval.runners.models import S3Config
from olmo_eval.storage import StorageBackend

console = Console()


class RunnerFactory:
    """Factory for creating evaluation runners based on configuration."""

    def __init__(
        self,
        config: RunConfig,
        storages: list[StorageBackend],
        s3_config: S3Config | None = None,
    ):
        """Initialize the factory.

        Args:
            config: Parsed run configuration.
            storages: List of initialized storage backends.
            s3_config: Optional S3 configuration.
        """
        self.config = config
        self.storages = storages
        self.s3_config = s3_config

    def create_agent_runner(self) -> Any:
        """Create an AgentEvalRunner for agent tasks.

        Returns:
            Configured AgentEvalRunner instance.
        """
        from olmo_eval.runners import AgentEvalRunner

        console.print("[bold cyan]Using AgentEvalRunner[/bold cyan]")

        model_name = self.config.model_names[0]
        return AgentEvalRunner(
            model_name=model_name,
            task_specs=self.config.task_specs,
            output_dir=self.config.output_dir,
            storages=self.storages,
            num_gpus=self.config.num_gpus,
            task_overrides=self.config.task_overrides,
            model_overrides=self.config.per_model_overrides.get(model_name, {}),
            s3_config=self.s3_config,
            experiment_name=self.config.experiment_name,
            experiment_group=self.config.experiment_group,
            alias=self.config.alias,
            save_predictions=self.config.save_predictions,
            save_requests=self.config.save_requests,
            inspect_instance=self.config.inspect_instance,
            inspect_formatted=self.config.inspect_formatted,
            inspect_tokens=self.config.inspect_tokens,
            inspect_response=self.config.inspect_response,
            inspect_request=self.config.inspect_request,
        )

    def create_async_runner(self) -> Any:
        """Create an AsyncEvalRunner for parallel task execution.

        Returns:
            Configured AsyncEvalRunner instance.
        """
        from olmo_eval.runners.simple import AsyncEvalRunner

        console.print("[bold cyan]Using AsyncEvalRunner[/bold cyan]")

        return AsyncEvalRunner(
            model_names=self.config.model_names,
            task_specs=self.config.task_specs,
            output_dir=self.config.output_dir,
            provider_override=self.config.provider,
            storages=self.storages,
            num_workers=self.config.num_workers,
            gpus_per_worker=self.config.gpus_per_worker,
            attention_backend=self.config.attention_backend.upper()
            if self.config.attention_backend
            else None,
            task_overrides=self.config.task_overrides,
            model_overrides=self.config.per_model_overrides,
            s3_config=self.s3_config,
            experiment_name=self.config.experiment_name,
            experiment_group=self.config.experiment_group,
            alias=self.config.alias,
            save_predictions=self.config.save_predictions,
            save_requests=self.config.save_requests,
            inspect_instance=self.config.inspect_instance,
            inspect_formatted=self.config.inspect_formatted,
            inspect_tokens=self.config.inspect_tokens,
            inspect_response=self.config.inspect_response,
            inspect_request=self.config.inspect_request,
        )

    def create(self) -> Any:
        """Create the appropriate runner based on configuration.

        Returns:
            Configured runner instance (agent or async).
        """
        if self.config.runner_type == RunnerType.AGENT:
            return self.create_agent_runner()
        else:
            return self.create_async_runner()
