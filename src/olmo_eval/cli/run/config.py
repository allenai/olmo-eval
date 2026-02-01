"""Configuration building for the run command."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class RunConfig:
    """Parsed and validated configuration for an evaluation run."""

    # Model configuration
    model_names: list[str]
    per_model_overrides: dict[str, dict[str, Any]]

    # Task configuration
    task_specs: list[str]
    task_overrides: dict[str, dict[str, Any]]

    # Runner configuration
    output_dir: str
    provider: str | None = None
    attention_backend: str | None = None
    use_async: bool = False
    use_async_stream: bool = False
    use_agent: bool = False
    num_workers: int | None = None
    gpus_per_worker: int = 1
    num_gpus: int = 1
    parallelism: int = 1

    # Storage configuration
    store: bool = False
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    s3_group: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "olmo_eval"
    db_user: str = "postgres"
    db_password: str = "postgres"

    # Experiment identification
    experiment_name: str | None = None
    experiment_group: str | None = None
    alias: str | None = None

    # Output options
    save_predictions: bool = True
    save_requests: bool = True

    # Debug/inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    # First model info (for single-model flows)
    first_model_name: str = ""
    first_model_overrides: dict[str, Any] = field(default_factory=dict)


class RunConfigBuilder:
    """Builds and validates run configuration from CLI arguments."""

    def __init__(
        self,
        models: tuple[str, ...],
        task: tuple[str, ...],
        output_dir: str,
        provider: str | None = None,
        attention_backend: str | None = None,
        use_async: bool = False,
        use_async_stream: bool = False,
        use_agent: bool = False,
        num_workers: int | None = None,
        gpus_per_worker: int = 1,
        num_gpus: int = 1,
        parallelism: int = 1,
        store: bool = False,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
        s3_group: str | None = None,
        s3_endpoint_url: str | None = None,
        s3_region: str = "us-east-1",
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "olmo_eval",
        db_user: str = "postgres",
        db_password: str = "postgres",
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        alias: str | None = None,
        save_predictions: bool = True,
        save_requests: bool = True,
        inspect_instance: bool = False,
        inspect_formatted: bool = False,
        inspect_tokens: bool = False,
        inspect_response: bool = False,
        inspect_request: bool = False,
    ):
        """Initialize the builder with raw CLI arguments."""
        self.models = models
        self.task = task
        self.output_dir = output_dir
        self.provider = provider
        self.attention_backend = attention_backend
        self.use_async = use_async
        self.use_async_stream = use_async_stream
        self.use_agent = use_agent
        self.num_workers = num_workers
        self.gpus_per_worker = gpus_per_worker
        self.num_gpus = num_gpus
        self.parallelism = parallelism
        self.store = store
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_group = s3_group
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_region = s3_region
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.experiment_name = experiment_name
        self.experiment_group = experiment_group
        self.alias = alias
        self.save_predictions = save_predictions
        self.save_requests = save_requests
        self.inspect_instance = inspect_instance
        self.inspect_formatted = inspect_formatted
        self.inspect_tokens = inspect_tokens
        self.inspect_response = inspect_response
        self.inspect_request = inspect_request

    def build(self) -> RunConfig:
        """Parse inputs and build configuration.

        Returns:
            RunConfig with parsed and validated settings.
        """
        from olmo_eval.cli.utils import parse_model_spec, parse_task_spec_with_overrides

        # Parse model specs to extract inline overrides
        parsed_models: list[tuple[str, dict[str, Any]]] = [parse_model_spec(m) for m in self.models]

        # Parse task specs to extract inline overrides
        task_overrides: dict[str, dict[str, Any]] = {}
        task_specs: list[str] = []
        for t in self.task:
            spec_without_overrides, overrides = parse_task_spec_with_overrides(t)
            task_specs.append(spec_without_overrides)
            if overrides:
                task_overrides[spec_without_overrides] = overrides

        # Extract model names and per-model overrides
        model_names = [name for name, _overrides in parsed_models]
        per_model_overrides = {name: overrides for name, overrides in parsed_models if overrides}

        # Get first model info
        first_model_name, first_model_overrides = parsed_models[0] if parsed_models else ("", {})

        # Apply model-level provider/attention_backend overrides
        provider = self.provider
        attention_backend = self.attention_backend
        if not provider and "provider" in first_model_overrides:
            provider = first_model_overrides["provider"]
        if not attention_backend and "attention_backend" in first_model_overrides:
            attention_backend = first_model_overrides["attention_backend"]

        return RunConfig(
            model_names=model_names,
            per_model_overrides=per_model_overrides,
            task_specs=task_specs,
            task_overrides=task_overrides,
            output_dir=self.output_dir,
            provider=provider,
            attention_backend=attention_backend,
            use_async=self.use_async,
            use_async_stream=self.use_async_stream,
            use_agent=self.use_agent,
            num_workers=self.num_workers,
            gpus_per_worker=self.gpus_per_worker,
            num_gpus=self.num_gpus,
            parallelism=self.parallelism,
            store=self.store,
            s3_bucket=self.s3_bucket,
            s3_prefix=self.s3_prefix,
            s3_group=self.s3_group,
            s3_endpoint_url=self.s3_endpoint_url,
            s3_region=self.s3_region,
            db_host=self.db_host,
            db_port=self.db_port,
            db_name=self.db_name,
            db_user=self.db_user,
            db_password=self.db_password,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            alias=self.alias,
            save_predictions=self.save_predictions,
            save_requests=self.save_requests,
            inspect_instance=self.inspect_instance,
            inspect_formatted=self.inspect_formatted,
            inspect_tokens=self.inspect_tokens,
            inspect_response=self.inspect_response,
            inspect_request=self.inspect_request,
            first_model_name=first_model_name,
            first_model_overrides=first_model_overrides,
        )

    def validate_flags(self) -> bool:
        """Validate CLI flag combinations and print warnings.

        Returns:
            True if validation passes, raises SystemExit on fatal errors.
        """
        # Warning for num-workers without async
        if self.num_workers is not None and not self.use_async and not self.use_async_stream:
            console.print(
                "[yellow]Warning:[/yellow] --num-workers has no effect without "
                "--async or --async-stream"
            )

        if self.gpus_per_worker != 1 and not self.use_async and not self.use_async_stream:
            console.print(
                "[yellow]Warning:[/yellow] --gpus-per-worker has no effect without "
                "--async or --async-stream"
            )

        # Warning for conflicting flags
        if self.use_async and self.use_async_stream:
            console.print(
                "[yellow]Warning:[/yellow] Both --async and --async-stream specified. "
                "Using --async-stream."
            )

        # Warning for provider override with async-stream
        if self.use_async_stream and self.provider and self.provider != "vllm":
            console.print(
                f"[yellow]Warning:[/yellow] --async-stream only supports vLLM provider, "
                f"ignoring --provider={self.provider}"
            )

        # Check for incompatible flags with --agent
        if self.use_agent:
            if self.use_async or self.use_async_stream:
                console.print(
                    "[red]Error:[/red] --agent cannot be used with --async or --async-stream"
                )
                raise SystemExit(1)
            if len(self.models) > 1:
                console.print(
                    "[red]Error:[/red] --agent only supports a single model. "
                    "Use beaker launch for multi-model agent runs."
                )
                raise SystemExit(1)

        return True

    def validate_bpb_tasks(self, task_specs: list[str]) -> None:
        """Check for incompatible task types with --async-stream.

        Args:
            task_specs: List of task specifications to check.

        Raises:
            SystemExit: If BPB tasks are used with --async-stream.
        """
        if not self.use_async_stream:
            return

        bpb_tasks = [t for t in task_specs if ":bpb" in t]
        if bpb_tasks:
            console.print(
                "\n[bold red]Error:[/bold red] The following :bpb tasks cannot run "
                "with --async-stream:\n"
                f"  {', '.join(bpb_tasks)}\n\n"
                "[yellow]BPB (bits-per-byte) tasks use loglikelihood scoring which "
                "requires\n"
                "prompt_logprobs - a feature not supported by the streaming vLLM "
                "backend.[/yellow]\n\n"
                "Use [bold]--async[/bold] or the default sequential mode instead:\n"
                f"  olmo-eval run -m <model> -t {' -t '.join(bpb_tasks)} --async\n"
            )
            raise SystemExit(1)
