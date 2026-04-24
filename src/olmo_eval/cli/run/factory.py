"""Runner factory for creating evaluation runners."""

from __future__ import annotations

from typing import Any

from olmo_eval.cli.run.config import RunConfig
from olmo_eval.runners.common.models import S3Config
from olmo_eval.storage import StorageBackend


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

    def create(self) -> Any:
        """Create the runner based on configuration.

        Returns:
            Configured AsyncEvalRunner instance.
        """
        from olmo_eval.runners.asynq.runner import AsyncEvalRunner

        # Get attention_backend from provider kwargs if specified
        provider_kwargs = self.config.harness_config.provider.kwargs
        attention_backend = provider_kwargs.get("attention_backend") if provider_kwargs else None

        return AsyncEvalRunner(
            harness_config=self.config.harness_config,
            task_specs=self.config.task_specs,
            output_dir=self.config.output_dir,
            storages=self.storages,
            attention_backend=attention_backend,
            task_overrides=self.config.task_overrides,
            s3_config=self.s3_config,
            experiment_name=self.config.experiment_name,
            experiment_group=self.config.experiment_group,
            save_predictions=self.config.save_predictions,
            save_requests=self.config.save_requests,
            inspect_instance=self.config.inspect_instance,
            inspect_formatted=self.config.inspect_formatted,
            inspect_tokens=self.config.inspect_tokens,
            inspect_response=self.config.inspect_response,
            inspect_request=self.config.inspect_request,
        )
