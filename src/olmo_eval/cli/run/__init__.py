"""Run command for olmo-eval CLI.

This module provides the main 'run' command for executing evaluations.
Configuration parsing, storage setup, and runner creation are delegated to:
- config.py: RunConfigBuilder for parsing and validating CLI arguments
- storage.py: StorageSetup for initializing storage backends
- factory.py: RunnerFactory for creating appropriate runners
- options.py: Option decorators for logical grouping
"""

import click

from olmo_eval.cli.run.options import (
    experiment_options,
    harness_options,
    inspect_options,
    output_options,
    parallelism_options,
    storage_options,
)
from olmo_eval.cli.utils import (
    OrderedMultiOption,
    console,
    print_runtime_environment,
    process_ordered_args,
    reconstruct_ordered_args,
)


@click.command()
# Core options (inline)
@click.option(
    "--model",
    "-m",
    required=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Model name or preset. Use --override after to add overrides.",
)
@click.option(
    "--task",
    "-t",
    multiple=True,
    required=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Task spec or suite. Use --override after to add overrides.",
)
@click.option(
    "--override",
    "-o",
    "cli_override",
    multiple=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Override for preceding -m or -t (e.g., -o provider.kind=vllm -o limit=100)",
)
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
# Grouped options via decorators
@harness_options
@parallelism_options
@storage_options
@experiment_options
@output_options
@inspect_options
def run(
    # Core options
    model: str,
    task: tuple[str, ...],
    cli_override: tuple[str, ...],
    config: str | None,
    # Harness options
    harness_preset: str | None,
    harness_config: str | None,
    # Parallelism options
    num_gpus: int,
    parallelism: int,
    # Storage options
    store: bool,
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
    # Experiment options
    experiment_name: str | None,
    experiment_group: str | None,
    # Output options
    output_dir: str,
    save_predictions: bool,
    save_requests: bool,
    dry_run: bool,
    # Inspect options
    debug_requests: bool,
    debug_provider: bool,
    inspect: bool,
    inspect_instance: bool,
    inspect_formatted: bool,
    inspect_tokens: bool,
    inspect_response: bool,
    inspect_request: bool,
) -> None:
    """Run evaluation on specified tasks.

    Use -o/--override after -m or -t to apply overrides:

        olmo-eval run -m llama3.1-8b -o provider.kind=vllm -t mmlu -o limit=100
    """
    import os
    import sys

    from olmo_eval.cli.run.config import RunConfigBuilder
    from olmo_eval.cli.run.factory import RunnerFactory
    from olmo_eval.cli.run.storage import StorageSetup
    from olmo_eval.common.logging import configure_logging
    from olmo_eval.runners import ValidationError

    # Process ordered args to associate overrides with tasks/harness
    ordered_args = reconstruct_ordered_args(sys.argv[1:])
    task_overrides, harness_overrides = process_ordered_args(ordered_args)

    # Configure logging for Beaker job visibility
    configure_logging(level="INFO")

    # Set debug environment variables
    if debug_requests:
        os.environ["OLMO_EVAL_DEBUG_REQUESTS"] = "1"
        os.environ["VLLM_DEBUG_REQUESTS"] = "1"
    if debug_provider:
        os.environ["OLMO_EVAL_DEBUG_PROVIDER"] = "1"

    # Expand --inspect to enable all individual inspect flags
    if inspect:
        inspect_instance = True
        inspect_formatted = True
        inspect_tokens = True
        inspect_response = True
        inspect_request = True

    # Print runtime environment summary
    print_runtime_environment()

    # Build configuration
    config_builder = RunConfigBuilder(
        model=model,
        task=task,
        output_dir=output_dir,
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
        save_predictions=save_predictions,
        save_requests=save_requests,
        inspect_instance=inspect_instance,
        inspect_formatted=inspect_formatted,
        inspect_tokens=inspect_tokens,
        inspect_response=inspect_response,
        inspect_request=inspect_request,
        cli_task_overrides=task_overrides,
        harness_preset=harness_preset,
        harness_config_path=harness_config,
        cli_harness_overrides=harness_overrides,
    )

    from rich.panel import Panel
    from rich.pretty import Pretty

    # Build configuration
    run_config = config_builder.build()

    # Print the config with resolved tasks
    console.print(
        Panel(
            Pretty(run_config, expand_all=True),
            title="[bold]Run Configuration[/bold]",
            border_style="cyan",
        )
    )

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
            console.print_exception()
            raise SystemExit(1) from None
