"""Runner factory for creating evaluation runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from olmo_eval.cli.run.config import RunConfig
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

        return AgentEvalRunner(
            model_name=self.config.first_model_name,
            task_specs=self.config.task_specs,
            output_dir=self.config.output_dir,
            storages=self.storages,
            num_gpus=self.config.num_gpus,
            task_overrides=self.config.task_overrides,
            model_overrides=self.config.first_model_overrides,
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

    def create_streaming_runner(self) -> Any:
        """Create a StreamingEvalRunner for async streaming.

        Returns:
            Configured StreamingEvalRunner instance.
        """
        from olmo_eval.runners.simple import StreamingEvalRunner

        console.print("[bold cyan]Using StreamingEvalRunner[/bold cyan]")

        return StreamingEvalRunner(
            model_names=self.config.model_names,
            task_specs=self.config.task_specs,
            output_dir=self.config.output_dir,
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

    def create_sync_runner(
        self,
        model_name: str,
        model_overrides: dict[str, Any],
    ) -> Any:
        """Create a SyncEvalRunner for sequential execution.

        Args:
            model_name: Model name for this runner.
            model_overrides: Model-specific overrides.

        Returns:
            Configured SyncEvalRunner instance.
        """
        from olmo_eval.runners import SyncEvalRunner

        # Apply per-model provider overrides
        effective_provider = model_overrides.get("provider", self.config.provider)
        effective_attention_backend = model_overrides.get(
            "attention_backend", self.config.attention_backend
        )

        return SyncEvalRunner(
            model_name=model_name,
            task_specs=self.config.task_specs,
            output_dir=self.config.output_dir,
            provider_override=effective_provider,
            storages=self.storages,
            attention_backend=effective_attention_backend.upper()
            if effective_attention_backend
            else None,
            task_overrides=self.config.task_overrides,
            model_overrides=model_overrides,
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
            Configured runner instance (agent, streaming, async, or sync).
        """
        if self.config.use_agent:
            return self.create_agent_runner()
        elif self.config.use_async_stream:
            return self.create_streaming_runner()
        elif self.config.use_async:
            return self.create_async_runner()
        else:
            # For sync mode, return first sync runner (multi-model handled separately)
            return self.create_sync_runner(
                self.config.first_model_name,
                self.config.first_model_overrides,
            )

    def run_sequential_models(self, dry_run: bool = False) -> None:
        """Run all models sequentially for sync mode.

        Args:
            dry_run: If True, print config instead of running.
        """
        from olmo_eval.runners import ValidationError

        model_names = self.config.model_names
        if len(model_names) > 1:
            console.print(f"[bold cyan]Running {len(model_names)} models sequentially[/bold cyan]")

        # Re-parse to get per-model overrides (since we need both name and overrides together)
        for i, model_name in enumerate(model_names):
            model_overrides = self.config.per_model_overrides.get(model_name, {})

            if len(model_names) > 1:
                console.print(f"\n[bold]Model {i + 1}/{len(model_names)}:[/bold] {model_name}")

            runner = self.create_sync_runner(model_name, model_overrides)

            try:
                runner.validate()
            except ValidationError as e:
                console.print(f"[red]Validation error:[/red]\n{e}")
                raise SystemExit(1) from None

            if dry_run:
                runner.print_config()
            else:
                try:
                    runner.run()
                except Exception as e:
                    console.print(f"\n[bold red]Evaluation failed:[/bold red] {e}")
                    raise SystemExit(1) from None
