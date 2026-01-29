"""Run command for olmo-eval CLI."""

from typing import Any

import click

from olmo_eval.cli.utils import (
    console,
    parse_model_spec,
    parse_task_spec_with_overrides,
    print_runtime_environment,
)
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR


@click.command()
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    required=True,
    help="Model name or preset. Can specify multiple times for multi-model runs.",
)
@click.option("--task", "-t", multiple=True, required=True, help="Task spec or suite")
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--output-dir", "-o", default=BEAKER_RESULT_DIR, help="Output directory")
@click.option("--num-shots", type=int, help="Override num_fewshot for all tasks")
@click.option("--limit", type=int, help="Override instance limit for all tasks")
@click.option("--temperature", type=float, help="Override temperature for all tasks")
@click.option("--provider", type=click.Choice(["hf", "vllm", "litellm"]), help="Override provider")
@click.option(
    "--store",
    is_flag=True,
    help="Persist results to the configured database",
)
@click.option("--dry-run", is_flag=True, help="Print config and exit without running")
@click.option(
    "--async",
    "use_async",
    is_flag=True,
    help="Use async runner for parallel task execution",
)
@click.option(
    "--async-stream",
    "use_async_stream",
    is_flag=True,
    help="Use streaming async runner with vLLM's AsyncLLMEngine for true continuous batching",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of workers for async mode (default: auto-detect from GPUs)",
)
@click.option(
    "--gpus-per-worker",
    type=int,
    default=1,
    help="Number of GPUs each worker uses (default: 1)",
)
@click.option(
    "--attention-backend",
    type=click.Choice(["FLASHINFER", "FLASH_ATTN"], case_sensitive=False),
    default=None,
    help="vLLM attention backend (e.g., FLASHINFER for better performance on supported GPUs)",
)
@click.option(
    "--parallelism",
    "-P",
    type=int,
    default=1,
    help="Number of model instances to run in parallel (passed from launch command)",
)
@click.option(
    "--s3-bucket",
    help="S3 bucket for storing evaluation results",
)
@click.option(
    "--s3-prefix",
    help="S3 prefix/path within bucket for results",
)
@click.option(
    "--s3-group",
    help="S3 group name (used in path structure)",
)
@click.option(
    "--s3-endpoint-url",
    help="S3 endpoint URL (for S3-compatible storage)",
)
@click.option(
    "--s3-region",
    default="us-east-1",
    help="S3 region (default: us-east-1)",
)
@click.option(
    "--db-host",
    envvar="PGHOST",
    default="localhost",
    help="PostgreSQL host",
)
@click.option(
    "--db-port",
    type=int,
    envvar="PGPORT",
    default=5432,
    help="PostgreSQL port",
)
@click.option(
    "--db-name",
    envvar="PGDATABASE",
    default="olmo_eval",
    help="PostgreSQL database name",
)
@click.option(
    "--db-user",
    envvar="PGUSER",
    default="postgres",
    help="PostgreSQL user",
)
@click.option(
    "--db-password",
    envvar="PGPASSWORD",
    default="postgres",
    help="PostgreSQL password",
)
@click.option(
    "--experiment-name",
    help="Human-readable experiment name for database storage",
)
@click.option(
    "--experiment-group",
    help="Experiment group for grouping related experiments (defaults to experiment-name)",
)
@click.option(
    "--alias",
    "-a",
    help="Short name for model (used as model_name in DB, original path stored as model_path)",
)
def run(
    models: tuple[str, ...],
    task: tuple[str, ...],
    config: str | None,
    output_dir: str,
    num_shots: int | None,
    limit: int | None,
    temperature: float | None,
    provider: str | None,
    store: bool,
    dry_run: bool,
    use_async: bool,
    use_async_stream: bool,
    num_workers: int | None,
    gpus_per_worker: int,
    attention_backend: str | None,
    parallelism: int,
    s3_bucket: str | None,
    s3_prefix: str | None,
    s3_group: str | None,
    s3_endpoint_url: str | None,
    s3_region: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    experiment_name: str | None,
    experiment_group: str | None,
    alias: str | None,
) -> None:
    """Run evaluation on specified tasks.

    Supports multiple models: use -m multiple times for multi-model runs.
    With --async, runs all (model, task) pairs with per-model workers.
    With --async-stream, uses vLLM's AsyncLLMEngine for true continuous batching.
    Without --async or --async-stream, runs sequentially for each model.

    Inline overrides can be specified in -m and -t flags:
        -m model::provider=vllm,tokenizer=allenai/dolma2-tokenizer
        -t task:olmes::temperature=0.6,num_fewshot=5
    """
    from olmo_eval.core.logging import configure_logging
    from olmo_eval.runners import SyncEvalRunner, ValidationError

    # Configure logging for Beaker job visibility
    configure_logging(level="INFO")

    # Print runtime environment summary
    print_runtime_environment()

    # Parse model specs to extract inline overrides
    parsed_models: list[tuple[str, dict[str, Any]]] = [parse_model_spec(m) for m in models]

    # Parse task specs to extract inline overrides
    task_overrides: dict[str, dict[str, Any]] = {}
    task_specs: list[str] = []
    for t in task:
        spec_without_overrides, overrides = parse_task_spec_with_overrides(t)
        task_specs.append(spec_without_overrides)
        if overrides:
            task_overrides[spec_without_overrides] = overrides

    # Extract model-level overrides
    first_model_name, first_model_overrides = parsed_models[0] if parsed_models else ("", {})

    # Model overrides can specify provider/attention_backend
    if not provider and "provider" in first_model_overrides:
        provider = first_model_overrides["provider"]
    if not attention_backend and "attention_backend" in first_model_overrides:
        attention_backend = first_model_overrides["attention_backend"]

    # Warning for num-workers without async
    if num_workers is not None and not use_async and not use_async_stream:
        console.print(
            "[yellow]Warning:[/yellow] --num-workers has no effect without "
            "--async or --async-stream"
        )

    if gpus_per_worker != 1 and not use_async and not use_async_stream:
        console.print(
            "[yellow]Warning:[/yellow] --gpus-per-worker has no effect without "
            "--async or --async-stream"
        )

    # Warning for conflicting flags
    if use_async and use_async_stream:
        console.print(
            "[yellow]Warning:[/yellow] Both --async and --async-stream specified. "
            "Using --async-stream."
        )
        use_async = False

    # Warning for provider override with async-stream
    if use_async_stream and provider and provider != "vllm":
        console.print(
            f"[yellow]Warning:[/yellow] --async-stream only supports vLLM provider, "
            f"ignoring --provider={provider}"
        )

    # Set up storage backend if enabled
    storages: list = []
    if store:
        from olmo_eval.storage import get_backend

        try:
            storage = get_backend(
                "postgres",
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
            )
            storage.initialize()
            storages.append(storage)
            console.print(
                f"[green]Connected to postgres storage:[/green] {db_host}:{db_port}/{db_name}"
            )
        except ImportError as e:
            console.print(f"[red]Storage backend error:[/red] {e}")
            raise SystemExit(1) from None
        except Exception as e:
            console.print(f"[red]Failed to initialize storage backend:[/red] {e}")
            raise SystemExit(1) from None

    # Set up S3 config if specified
    s3_config = None
    if s3_bucket or s3_prefix or s3_group:
        # Validate that all required S3 options are provided
        if not s3_bucket:
            console.print("[red]Error:[/red] --s3-bucket is required for S3 uploads")
            raise SystemExit(1)
        if not s3_prefix:
            console.print("[red]Error:[/red] --s3-prefix is required for S3 uploads")
            raise SystemExit(1)
        if not s3_group:
            console.print("[red]Error:[/red] --s3-group is required for S3 uploads")
            raise SystemExit(1)

        from olmo_eval.runners.mixins import S3Config

        s3_config = S3Config(
            bucket=s3_bucket,
            prefix=s3_prefix,
            group=s3_group,
            endpoint_url=s3_endpoint_url,
            region=s3_region,
        )
        console.print(
            f"[green]S3 uploads enabled:[/green] s3://{s3_bucket}/{s3_prefix}/{s3_group}/..."
        )

    # Check for incompatible task types with --async-stream
    if use_async_stream:
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

    # Extract model names and build per-model overrides dict
    model_names = [name for name, _overrides in parsed_models]
    per_model_overrides = {name: overrides for name, overrides in parsed_models if overrides}

    # Choose runner based on --async or --async-stream flag
    if use_async_stream:
        from olmo_eval.runners.asynchronous import StreamingEvalRunner

        console.print("[bold cyan]Using StreamingEvalRunner[/bold cyan]")

        runner = StreamingEvalRunner(
            model_names=model_names,
            task_specs=task_specs,
            output_dir=output_dir,
            num_shots_override=num_shots,
            limit_override=limit,
            temperature=temperature,
            storages=storages,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
            attention_backend=attention_backend.upper() if attention_backend else None,
            task_overrides=task_overrides,
            model_overrides=per_model_overrides,
            s3_config=s3_config,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            alias=alias,
        )
    elif use_async:
        from olmo_eval.runners.asynchronous import AsyncEvalRunner

        console.print("[bold cyan]Using AsyncEvalRunner[/bold cyan]")

        runner = AsyncEvalRunner(
            model_names=model_names,
            task_specs=task_specs,
            output_dir=output_dir,
            num_shots_override=num_shots,
            limit_override=limit,
            temperature=temperature,
            provider_override=provider,
            storages=storages,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
            attention_backend=attention_backend.upper() if attention_backend else None,
            task_overrides=task_overrides,
            model_overrides=per_model_overrides,
            s3_config=s3_config,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            alias=alias,
        )
    else:
        # Sequential runner - run each model in sequence
        if len(model_names) > 1:
            console.print(f"[bold cyan]Running {len(model_names)} models sequentially[/bold cyan]")

        # For sequential mode with multiple models, run each model separately
        for i, (model_name, model_overrides) in enumerate(parsed_models):
            if len(model_names) > 1:
                console.print(f"\n[bold]Model {i + 1}/{len(model_names)}:[/bold] {model_name}")

            # Apply per-model provider overrides
            effective_provider = model_overrides.get("provider", provider)
            effective_attention_backend = model_overrides.get(
                "attention_backend", attention_backend
            )

            runner = SyncEvalRunner(
                model_name=model_name,
                task_specs=task_specs,
                output_dir=output_dir,
                num_shots_override=num_shots,
                limit_override=limit,
                temperature=temperature,
                provider_override=effective_provider,
                storages=storages,
                attention_backend=effective_attention_backend.upper()
                if effective_attention_backend
                else None,
                task_overrides=task_overrides,
                model_overrides=model_overrides,
                s3_config=s3_config,
                experiment_name=experiment_name,
                experiment_group=experiment_group,
                alias=alias,
            )

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

        return  # Exit early since we handled everything in the loop

    # Validate inputs before running (applies to both dry-run and actual runs)
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
