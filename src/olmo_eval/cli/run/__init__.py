"""Run command for olmo-eval CLI.

This module provides the main 'run' command for executing evaluations.
Configuration parsing, storage setup, and runner creation are delegated to:
- config.py: RunConfigBuilder for parsing and validating CLI arguments
- storage.py: StorageSetup for initializing storage backends
- factory.py: RunnerFactory for creating appropriate runners
"""

import click

from olmo_eval.cli.utils import console, print_runtime_environment
from olmo_eval.core.constants.infrastructure import LOCAL_RESULT_DIR


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
@click.option("--output-dir", "-o", default=LOCAL_RESULT_DIR, help="Output directory")
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
@click.option(
    "--agent",
    is_flag=True,
    help="Use agent runner for multi-turn agent tasks (e.g., simpleqa_agent)",
)
@click.option(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs for tensor parallelism (used with --agent)",
)
@click.option(
    "--debug-requests",
    is_flag=True,
    help="Log HTTP requests/responses to inference providers",
)
@click.option(
    "--debug-provider",
    is_flag=True,
    help="Enable verbose provider logging",
)
@click.option(
    "--save-predictions/--no-save-predictions",
    "save_predictions",
    default=True,
    help="Save per-instance predictions to JSONL (default: enabled)",
)
@click.option(
    "--save-requests/--no-save-requests",
    "save_requests",
    default=True,
    help="Save per-instance requests to JSONL (default: enabled)",
)
@click.option(
    "--inspect-instance",
    is_flag=True,
    help="Print the first instance of each task before running evaluation",
)
@click.option(
    "--inspect-formatted",
    is_flag=True,
    help="Show formatted prompt (after template applied) before evaluation",
)
@click.option(
    "--inspect-tokens",
    is_flag=True,
    help="Show token array before evaluation",
)
@click.option(
    "--inspect-response",
    is_flag=True,
    help="Print the first response of each task after model generation",
)
@click.option(
    "--inspect-request",
    is_flag=True,
    help="Print the first request of each task before model generation",
)
def run(
    models: tuple[str, ...],
    task: tuple[str, ...],
    config: str | None,
    output_dir: str,
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
    agent: bool,
    num_gpus: int,
    debug_requests: bool,
    debug_provider: bool,
    save_predictions: bool,
    save_requests: bool,
    inspect_instance: bool,
    inspect_formatted: bool,
    inspect_tokens: bool,
    inspect_response: bool,
    inspect_request: bool,
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
    import os

    from olmo_eval.cli.run.config import RunConfigBuilder
    from olmo_eval.cli.run.factory import RunnerFactory
    from olmo_eval.cli.run.storage import StorageSetup
    from olmo_eval.core.logging import configure_logging
    from olmo_eval.runners import ValidationError

    # Configure logging for Beaker job visibility
    configure_logging(level="INFO")

    # Set debug environment variables
    if debug_requests:
        os.environ["OLMO_EVAL_DEBUG_REQUESTS"] = "1"
    if debug_provider:
        os.environ["OLMO_EVAL_DEBUG_PROVIDER"] = "1"

    # Print runtime environment summary
    print_runtime_environment()

    # Handle conflicting async flags
    effective_use_async = use_async
    if use_async and use_async_stream:
        effective_use_async = False

    # Build configuration
    config_builder = RunConfigBuilder(
        models=models,
        task=task,
        output_dir=output_dir,
        provider=provider,
        attention_backend=attention_backend,
        use_async=effective_use_async,
        use_async_stream=use_async_stream,
        use_agent=agent,
        num_workers=num_workers,
        gpus_per_worker=gpus_per_worker,
        num_gpus=num_gpus,
        parallelism=parallelism,
        store=store,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_group=s3_group,
        s3_endpoint_url=s3_endpoint_url,
        s3_region=s3_region,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        alias=alias,
        save_predictions=save_predictions,
        save_requests=save_requests,
        inspect_instance=inspect_instance,
        inspect_formatted=inspect_formatted,
        inspect_tokens=inspect_tokens,
        inspect_response=inspect_response,
        inspect_request=inspect_request,
    )

    # Validate CLI flags
    config_builder.validate_flags()

    # Build configuration
    run_config = config_builder.build()

    # Validate BPB tasks with async-stream
    config_builder.validate_bpb_tasks(run_config.task_specs)

    # Set up storage backends
    storage_setup = StorageSetup(
        store=store,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_group=s3_group,
        s3_endpoint_url=s3_endpoint_url,
        s3_region=s3_region,
    )
    storages, s3_config = storage_setup.setup()

    # Create runner factory
    factory = RunnerFactory(run_config, storages, s3_config)

    # Handle sequential multi-model sync mode separately
    if not agent and not use_async_stream and not effective_use_async:
        factory.run_sequential_models(dry_run=dry_run)
        return

    # Create and run the appropriate runner
    runner = factory.create()

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
