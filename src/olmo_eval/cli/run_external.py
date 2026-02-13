"""CLI command for running external evaluations."""

from __future__ import annotations

import click

from olmo_eval.cli.utils import console
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR


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
    "--list-evals",
    is_flag=True,
    help="List available external evaluations and exit",
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
    list_evals: bool,
) -> None:
    """Run external black-box evaluations.

    External evaluations run inside sandbox containers and communicate with
    the model via an OpenAI-compatible API.

    Examples:

        # Run tau2_bench on a model
        olmo-eval run-external -m meta-llama/Llama-3.1-8B-Instruct -e tau2_bench

        # Run with custom output directory
        olmo-eval run-external -m my-model -e tau2_bench -O ./results

        # Run multiple evaluations
        olmo-eval run-external -m my-model -e tau2_bench -e other_bench
    """
    from olmo_eval.common.logging import configure_logging
    from olmo_eval.evals.external import list_external_evals

    configure_logging(level="INFO")

    # Handle --list-evals flag
    if list_evals:
        available = list_external_evals()
        if not available:
            console.print("[dim]No external evaluations registered.[/dim]")
        else:
            console.print("[bold]Available external evaluations:[/bold]")
            for name in available:
                console.print(f"  - {name}")
        return

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

    # Apply overrides
    provider_config = provider_config.with_overrides(
        kind=provider,
        base_url=base_url,
        tensor_parallel_size=tensor_parallel_size if tensor_parallel_size > 1 else None,
    )

    # Create runner
    from olmo_eval.runners.external import ExternalEvalRunner

    runner = ExternalEvalRunner(
        provider_config=provider_config,
        external_eval_names=list(evals),
        output_dir=output_dir,
        container_runtime=runtime,
        server_port=port,
    )

    # Validate
    try:
        runner.validate()
    except ValueError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise SystemExit(1) from None

    # Print configuration
    from rich.panel import Panel
    from rich.table import Table

    table = Table(title="External Evaluation Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Model", provider_config.model)
    table.add_row("Provider", provider_config.kind)
    table.add_row("Base URL", base_url or f"http://localhost:{port}")
    table.add_row("Evaluations", ", ".join(evals))
    table.add_row("Output Dir", output_dir)
    table.add_row("Container Runtime", runtime)
    if tensor_parallel_size > 1:
        table.add_row("Tensor Parallel Size", str(tensor_parallel_size))

    console.print(Panel(table))

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
