"""CLI command for running external evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import click

from olmo_eval.cli.utils import ConfiguredExternalEval, console
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
    default=1,
    help="Tensor parallel size for vLLM",
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
def run_external(
    model: str,
    evals: tuple[str, ...],
    output_dir: str,
    provider: str,
    base_url: str | None,
    tensor_parallel_size: int,
    port: int,
    runtime: str,
    dry_run: bool,
    eval_args: tuple[str, ...],
    provider_kwargs: tuple[str, ...],
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
    parsed_provider_kwargs: dict[str, Any] = {}
    for kwarg in provider_kwargs:
        if "=" not in kwarg:
            console.print(f"[red]Error:[/red] Invalid provider kwarg '{kwarg}', use key=value")
            raise SystemExit(1)
        key, value = kwarg.split("=", 1)
        # Type coercion for common types
        if value.lower() == "true":
            parsed_provider_kwargs[key] = True
        elif value.lower() == "false":
            parsed_provider_kwargs[key] = False
        elif value.isdigit():
            parsed_provider_kwargs[key] = int(value)
        elif value.replace(".", "", 1).isdigit():
            parsed_provider_kwargs[key] = float(value)
        else:
            parsed_provider_kwargs[key] = value

    # Apply overrides
    provider_config = provider_config.with_overrides(
        kind=provider,
        base_url=base_url,
        tensor_parallel_size=tensor_parallel_size if tensor_parallel_size > 1 else None,
        **parsed_provider_kwargs,
    )

    # Parse eval_args (supports both key=value and JSON dict format)
    parsed_args: dict[str, str] = {}
    for arg in eval_args:
        if arg.startswith("{"):
            try:
                parsed_args.update(json.loads(arg))
            except json.JSONDecodeError as e:
                console.print(f"[red]Error:[/red] Invalid JSON in -a argument: {e}")
                raise SystemExit(1) from None
        elif "=" in arg:
            key, value = arg.split("=", 1)
            parsed_args[key] = value
        else:
            console.print(f"[yellow]Warning:[/yellow] Invalid arg '{arg}', use key=value")

    # Create runner
    from olmo_eval.runners.external import ExternalEvalRunner

    runner = ExternalEvalRunner(
        provider_config=provider_config,
        external_eval_names=list(evals),
        output_dir=output_dir,
        container_runtime=runtime,
        server_port=port,
        eval_args=parsed_args,
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
            metrics = ", ".join(f"{k}={v:.4f}" for k, v in result.metrics.items())
        else:
            status = "[red]Failed[/red]"
            metrics = result.error or "Unknown error"

        results_table.add_row(name, status, metrics)

    console.print(results_table)

    # Exit with error if any evaluation failed
    if any(not r.success for r in results.values()):
        raise SystemExit(1)
