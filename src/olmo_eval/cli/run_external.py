"""CLI command for running external evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import click

from olmo_eval.cli.utils import ConfiguredExternalEval, console, parse_key_value_args
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR

if TYPE_CHECKING:
    from olmo_eval.inference.providers.config import ProviderConfig


@dataclass
class ExternalRunConfig:
    """Configuration for an external evaluation run."""

    provider: ProviderConfig
    evals: list[ConfiguredExternalEval]
    output_dir: str
    container_runtime: str
    server_port: int = 8000
    eval_args: dict[str, Any] = field(default_factory=dict)


@click.command(name="run-external")
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model name or path (HuggingFace ID or local path)",
)
@click.option(
    "--eval",
    "-e",
    "evals",
    multiple=True,
    required=True,
    help="External evaluation name(s) to run (can specify multiple)",
)
@click.option(
    "--output-dir",
    "-O",
    default=BEAKER_RESULT_DIR,
    help="Directory to write results",
)
@click.option(
    "--provider",
    "-p",
    default="vllm_server",
    type=click.Choice(["vllm", "vllm_server", "litellm"]),
    help="Inference provider to use",
)
@click.option(
    "--base-url",
    help="Base URL for the inference provider (if already running)",
)
@click.option(
    "--tensor-parallel-size",
    "--tp",
    type=int,
    default=None,
    help="Tensor parallel size for vLLM (overrides model preset)",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port for the vLLM server",
)
@click.option(
    "--runtime",
    type=click.Choice(["docker", "podman"]),
    default="podman",
    help="Container runtime to use for sandboxes",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print configuration without running",
)
@click.option(
    "--arg",
    "-a",
    "eval_args",
    multiple=True,
    help="Arguments for external evals (key=value or JSON dict format)",
)
@click.option(
    "--provider-kwarg",
    "-K",
    "provider_kwargs",
    multiple=True,
    help="Provider kwargs (key=value, e.g., -K enable_chunked_prefill=true)",
)
# Storage options
@click.option(
    "--store",
    is_flag=True,
    help="Persist results to the configured database",
)
@click.option("--s3-bucket", help="S3 bucket for storing evaluation results")
@click.option("--s3-prefix", help="S3 prefix/path within bucket for results")
@click.option("--s3-group", help="S3 group name (used in path structure)")
@click.option(
    "--s3-endpoint-url",
    envvar="S3_ENDPOINT_URL",
    help="S3 endpoint URL (for S3-compatible storage)",
)
@click.option(
    "--s3-region",
    default="us-east-1",
    envvar="AWS_REGION",
    help="S3 region (default: us-east-1)",
)
@click.option("--db-host", default="localhost", envvar="PGHOST", help="PostgreSQL host")
@click.option("--db-port", default=5432, type=int, envvar="PGPORT", help="PostgreSQL port")
@click.option("--db-name", default="olmo_eval", envvar="PGDATABASE", help="PostgreSQL database")
@click.option("--db-user", default="postgres", envvar="PGUSER", help="PostgreSQL user")
@click.option("--db-password", default="postgres", envvar="PGPASSWORD", help="PostgreSQL password")
# Experiment metadata
@click.option("--experiment-name", help="Human-readable experiment name")
@click.option("--experiment-group", help="Experiment group for grouping related experiments")
def run_external(
    model: str,
    evals: tuple[str, ...],
    output_dir: str,
    provider: str,
    base_url: str | None,
    tensor_parallel_size: int | None,
    port: int,
    runtime: str,
    dry_run: bool,
    eval_args: tuple[str, ...],
    provider_kwargs: tuple[str, ...],
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
    experiment_name: str | None,
    experiment_group: str | None,
) -> None:
    """Run external black-box evaluations.

    External evaluations run inside sandbox containers and communicate with
    the model via an OpenAI-compatible API.

    Examples:

        # Run tau2_bench on a model
        olmo-eval run-external -m meta-llama/Llama-3.1-8B-Instruct -e tau2_bench

        # Run with custom arguments (key=value format)
        olmo-eval run-external -m my-model -e tau2_bench -a domain=retail -a num_trials=10

        # Run with custom arguments (JSON format)
        olmo-eval run-external -m my-model -e tau2_bench -a '{"domain": "retail", "num_trials": 10}'

        # Run with custom output directory
        olmo-eval run-external -m my-model -e tau2_bench -O ./results

        # Run multiple evaluations
        olmo-eval run-external -m my-model -e tau2_bench -e other_bench
    """
    from olmo_eval.common.logging import configure_logging

    configure_logging(level="INFO")

    # Build provider config
    from olmo_eval.common.configs import get_provider_config

    try:
        provider_config = get_provider_config(model)
    except Exception:
        # Fall back to creating a basic config
        from olmo_eval.inference.providers.config import ProviderConfig

        provider_config = ProviderConfig(
            kind=provider,
            model=model,
        )

    # Parse provider kwargs (key=value format, with type coercion)
    try:
        parsed_provider_kwargs = parse_key_value_args(provider_kwargs, coerce_types=True)
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid provider kwarg: {e}")
        raise SystemExit(1) from None

    # Apply overrides
    provider_config = provider_config.with_overrides(
        kind=provider,
        base_url=base_url,
        tensor_parallel_size=tensor_parallel_size,
        **parsed_provider_kwargs,
    )

    # Parse eval_args (supports both key=value and JSON dict format)
    try:
        parsed_args = parse_key_value_args(eval_args, coerce_types=True)
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid eval arg: {e}")
        raise SystemExit(1) from None

    # Set up storage backends
    from olmo_eval.cli.run.storage import StorageSetup

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

    # Create runner
    from olmo_eval.runners.external import ExternalEvalRunner

    runner = ExternalEvalRunner(
        provider_config=provider_config,
        external_eval_names=list(evals),
        output_dir=output_dir,
        container_runtime=runtime,
        server_port=port,
        eval_args=parsed_args,
        s3_config=s3_config,
        storages=storages,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
    )

    # Validate
    try:
        runner.validate()
    except ValueError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise SystemExit(1) from None

    # Print configuration
    from rich.panel import Panel
    from rich.pretty import Pretty
    from rich.table import Table

    from olmo_eval.evals.external import get_external_eval

    configured_evals = [
        ConfiguredExternalEval.from_eval(get_external_eval(name), provider_config, parsed_args)
        for name in evals
    ]

    run_config = ExternalRunConfig(
        provider=provider_config,
        evals=configured_evals,
        output_dir=output_dir,
        container_runtime=runtime,
        server_port=port,
        eval_args=parsed_args,
    )

    console.print(
        Panel(
            Pretty(run_config, expand_all=True),
            title="[bold]Run Configuration[/bold]",
            border_style="cyan",
        )
    )

    if dry_run:
        console.print("\n[yellow]Dry run mode - not executing[/yellow]")
        return

    # Run evaluations
    console.print("\n[bold]Starting external evaluations...[/bold]")

    try:
        results = runner.run()
    except Exception as e:
        console.print(f"\n[bold red]Evaluation failed:[/bold red] {e}")
        console.print_exception()
        raise SystemExit(1) from None

    # Print summary
    console.print("\n[bold]Results Summary:[/bold]")

    results_table = Table()
    results_table.add_column("Evaluation", style="cyan")
    results_table.add_column("Status")
    results_table.add_column("Metrics")

    for name, result in results.items():
        if result.success:
            status = "[green]Success[/green]"
            # Format metrics vertically, one per line
            metrics_lines = []
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    if v == int(v):
                        metrics_lines.append(f"{k}: [bold]{int(v)}[/bold]")
                    else:
                        metrics_lines.append(f"{k}: [bold]{v:.4f}[/bold]")
                else:
                    metrics_lines.append(f"{k}: [bold]{v}[/bold]")
            metrics = "\n".join(metrics_lines)
        else:
            status = "[red]Failed[/red]"
            metrics = result.error or "Unknown error"

        results_table.add_row(name, status, metrics)

    console.print(results_table)

    # Exit with error if any evaluation failed
    if any(not r.success for r in results.values()):
        raise SystemExit(1)
