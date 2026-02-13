"""CLI command for running external evaluations."""

from __future__ import annotations

import json

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
    "--arg",
    "-a",
    "eval_args",
    multiple=True,
    help="Arguments for external evals (key=value or JSON dict format)",
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

    # Apply overrides
    provider_config = provider_config.with_overrides(
        kind=provider,
        base_url=base_url,
        tensor_parallel_size=tensor_parallel_size if tensor_parallel_size > 1 else None,
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
    if parsed_args:
        table.add_row("Eval Args", ", ".join(f"{k}={v}" for k, v in parsed_args.items()))

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
