"""olmo-eval CLI entry point."""

from dataclasses import dataclass
from datetime import UTC
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers suite registration
import olmo_eval.evals.tasks  # noqa: F401 - triggers task registration
from olmo_eval.core import get_model_presets
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR, DEFAULT_MAX_GPUS_PER_NODE
from olmo_eval.evals.suites import get_suite, list_suites
from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants
from olmo_eval.evals.tasks.core.registry import parse_overrides

# -----------------------------------------------------------------------------
# Dataclasses for pretty-printing launch configuration
# -----------------------------------------------------------------------------


@dataclass
class ModelSummary:
    """Summary of a model configuration."""

    name: str
    gpus: int = 1
    parallelism: int = 1
    alias: str | None = None
    backend: str | None = None
    overrides: dict[str, Any] | None = None  # Inline overrides from spec (::key=value)


@dataclass
class TaskSummary:
    """Summary of a task configuration for display."""

    name: str
    spec: str | None = None  # Original task spec (e.g., "humaneval:pass@1")
    variants: list[str] | None = None  # Applied variants/regimes
    formatter: Any = None
    scorers: tuple = ()
    metrics: tuple = ()
    num_fewshot: int = 0
    split: str = "test"
    primary_metric: str | None = None
    sampling_params: Any = None
    overrides: dict[str, Any] | None = None  # Inline overrides from spec (::key=value)


@dataclass
class RunnerConfig:
    """Runner configuration for display."""

    runner: type  # The runner class
    output_dir: str | None = None
    attention_backend: str | None = None
    # Async-only settings
    num_workers: int | str | None = None
    gpus_per_worker: int | None = None

    def __repr__(self) -> str:
        parts = [f"runner={self.runner.__name__}"]
        if self.output_dir is not None:
            parts.append(f"output_dir={self.output_dir!r}")
        if self.attention_backend is not None:
            parts.append(f"attention_backend={self.attention_backend!r}")
        if self.num_workers is not None:
            parts.append(f"num_workers={self.num_workers!r}")
        if self.gpus_per_worker is not None:
            parts.append(f"gpus_per_worker={self.gpus_per_worker}")
        return f"RunnerConfig({', '.join(parts)})"


@dataclass
class LaunchSummary:
    """Complete launch configuration summary for pretty-printing."""

    models: list[ModelSummary]
    tasks: list[TaskSummary]
    runner: RunnerConfig


console = Console()


# -----------------------------------------------------------------------------
# Spec parsing for inline overrides
# -----------------------------------------------------------------------------

# Keys that apply to TaskConfig
TASKCONFIG_KEYS = {"num_fewshot", "limit", "fewshot_seed"}

# Keys that apply to SamplingParams
SAMPLING_KEYS = {"temperature", "max_tokens", "top_p", "top_k", "num_samples"}

# Keys that apply to model/backend config
MODEL_KEYS = {
    "backend",
    "attention_backend",
    "gpus_per_worker",
    "tokenizer",
    "max_model_len",
    "load_format",
}


def parse_model_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse model spec into (model_name, overrides).

    Format: model[::key=value,...]

    Args:
        spec: Model specification string.

    Returns:
        Tuple of (model_name, overrides dict).

    Examples:
        >>> parse_model_spec("allenai/OLMo-7B")
        ("allenai/OLMo-7B", {})
        >>> parse_model_spec("allenai/OLMo-7B::backend=vllm")
        ("allenai/OLMo-7B", {"backend": "vllm"})
        >>> parse_model_spec("allenai/OLMo-7B::backend=vllm,attention_backend=FLASHINFER")
        ("allenai/OLMo-7B", {"backend": "vllm", "attention_backend": "FLASHINFER"})
    """
    main_part, _, override_str = spec.partition("::")
    overrides = parse_overrides(override_str) if override_str else {}
    return main_part, overrides


def parse_task_spec_with_overrides(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse task spec with inline overrides.

    Format: task[:variant...][::key=value,...]

    This extracts the overrides separately so they can be passed to the runner,
    while the task spec (without overrides) is used for task lookup.

    Args:
        spec: Task specification string.

    Returns:
        Tuple of (task_spec_without_overrides, overrides dict).

    Examples:
        >>> parse_task_spec_with_overrides("arc_easy:olmes")
        ("arc_easy:olmes", {})
        >>> parse_task_spec_with_overrides("gsm8k:olmes::temperature=0.6")
        ("gsm8k:olmes", {"temperature": 0.6})
        >>> parse_task_spec_with_overrides("gsm8k::temperature=0.6,max_tokens=512")
        ("gsm8k", {"temperature": 0.6, "max_tokens": 512})
    """
    spec_part, _, override_str = spec.partition("::")
    overrides = parse_overrides(override_str) if override_str else {}
    return spec_part, overrides


def _print_runtime_environment() -> None:
    """Print runtime environment summary for debugging."""
    import sys

    console.print("\n" + "=" * 60)
    console.print("RUNTIME ENVIRONMENT SUMMARY")
    console.print("=" * 60)
    console.print(f"Python:          {sys.version.split()[0]}")
    try:
        import torch  # type: ignore[import-not-found]

        console.print(f"PyTorch:         {torch.__version__}")
        console.print(f"CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"CUDA version:    {torch.version.cuda}")
            console.print(f"cuDNN version:   {torch.backends.cudnn.version()}")
            console.print(f"GPU count:       {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                console.print(f"  GPU {i}:         {torch.cuda.get_device_name(i)}")
    except ImportError:
        console.print("PyTorch:         NOT INSTALLED")
    try:
        import transformers

        console.print(f"Transformers:    {transformers.__version__}")
    except ImportError:
        console.print("Transformers:    NOT INSTALLED")
    try:
        import vllm  # type: ignore[import-not-found]

        console.print(f"vLLM:            {vllm.__version__}")
    except ImportError:
        console.print("vLLM:            NOT INSTALLED")
    console.print("=" * 60 + "\n")


@click.group()
def main() -> None:
    """olmo-eval command line interface."""
    pass


@main.group()
def beaker() -> None:
    """Beaker job management commands.

    Commands for launching, monitoring, and managing evaluation jobs on Beaker.
    """
    pass


@main.command()
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
@click.option("--backend", type=click.Choice(["hf", "vllm", "litellm"]), help="Override backend")
@click.option(
    "--storage-backend",
    "-s",
    "storage_backends",
    type=click.Choice(["s3", "postgres"]),
    multiple=True,
    help="Storage backend(s) for results. Can be specified multiple times.",
)
@click.option(
    "--storage-config",
    type=click.Path(exists=True),
    help="YAML config file for storage backend",
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
def run(
    models: tuple[str, ...],
    task: tuple[str, ...],
    config: str | None,
    output_dir: str,
    num_shots: int | None,
    limit: int | None,
    temperature: float | None,
    backend: str | None,
    storage_backends: tuple[str, ...],
    storage_config: str | None,
    dry_run: bool,
    use_async: bool,
    use_async_stream: bool,
    num_workers: int | None,
    gpus_per_worker: int,
    attention_backend: str | None,
    parallelism: int,
) -> None:
    """Run evaluation on specified tasks.

    Supports multiple models: use -m multiple times for multi-model runs.
    With --async, runs all (model, task) pairs with per-model workers.
    With --async-stream, uses vLLM's AsyncLLMEngine for true continuous batching.
    Without --async or --async-stream, runs sequentially for each model.

    Inline overrides can be specified in -m and -t flags:
        -m model::backend=vllm,tokenizer=allenai/dolma2-tokenizer
        -t task:olmes::temperature=0.6,num_fewshot=5
    """
    import logging

    from olmo_eval.runners.synchronous import SyncEvalRunner, ValidationError

    # Configure logging for Beaker job visibility
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    # Suppress noisy HuggingFace warnings
    import os

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Print runtime environment summary
    _print_runtime_environment()

    # Parse model specs to extract inline overrides
    parsed_models: list[tuple[str, dict[str, Any]]] = [parse_model_spec(m) for m in models]

    # Parse task specs to extract inline overrides
    # Store as dict mapping task_spec -> overrides
    task_overrides: dict[str, dict[str, Any]] = {}
    task_specs: list[str] = []
    for t in task:
        spec_without_overrides, overrides = parse_task_spec_with_overrides(t)
        task_specs.append(spec_without_overrides)
        if overrides:
            task_overrides[spec_without_overrides] = overrides

    # Extract model-level overrides (use first model's overrides for global settings)
    # In multi-model async mode, each model could have different settings
    first_model_name, first_model_overrides = parsed_models[0] if parsed_models else ("", {})

    # Model overrides can specify backend/attention_backend
    if not backend and "backend" in first_model_overrides:
        backend = first_model_overrides["backend"]
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

    # Warning for backend override with async-stream
    if use_async_stream and backend and backend != "vllm":
        console.print(
            f"[yellow]Warning:[/yellow] --async-stream only supports vLLM backend, "
            f"ignoring --backend={backend}"
        )

    # Set up storage backends if specified
    storages: list = []
    if storage_backends:
        from olmo_eval.storage import get_backend

        # Load storage config if provided
        storage_cfg = None
        if storage_config:
            from omegaconf import DictConfig, OmegaConf

            cfg = OmegaConf.load(storage_config)
            if isinstance(cfg, DictConfig):
                storage_cfg = cfg
            else:
                console.print("[red]Error:[/red] Storage config must be a YAML dict, not a list")
                raise SystemExit(1)

        for backend_name in storage_backends:
            # Get backend-specific config section
            storage_kwargs: dict = {}
            if storage_cfg:
                backend_cfg = storage_cfg.get(backend_name, {})
                storage_kwargs = OmegaConf.to_container(backend_cfg, resolve=True) or {}  # type: ignore

            try:
                storage = get_backend(backend_name, **storage_kwargs)
                storages.append(storage)
                console.print(f"[green]Initialized {backend_name} storage backend[/green]")
            except ImportError as e:
                console.print(f"[red]Storage backend error:[/red] {e}")
                raise SystemExit(1) from None
            except Exception as e:
                console.print(
                    f"[red]Failed to initialize {backend_name} storage backend:[/red] {e}"
                )
                raise SystemExit(1) from None

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
        console.print(f"[bold]Models:[/bold] {len(model_names)}")

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
        )
    elif use_async:
        from olmo_eval.runners.asynchronous import AsyncEvalRunner

        console.print("[bold cyan]Using AsyncEvalRunner[/bold cyan]")
        console.print(f"[bold]Models:[/bold] {len(model_names)}")

        runner = AsyncEvalRunner(
            model_names=model_names,
            task_specs=task_specs,
            output_dir=output_dir,
            num_shots_override=num_shots,
            limit_override=limit,
            temperature=temperature,
            backend_override=backend,
            storages=storages,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
            attention_backend=attention_backend.upper() if attention_backend else None,
            task_overrides=task_overrides,
            model_overrides=per_model_overrides,
        )
    else:
        # Sequential runner - run each model in sequence
        if len(model_names) > 1:
            console.print(f"[bold cyan]Running {len(model_names)} models sequentially[/bold cyan]")

        # For sequential mode with multiple models, run each model separately
        for i, (model_name, model_overrides) in enumerate(parsed_models):
            if len(model_names) > 1:
                console.print(f"\n[bold]Model {i + 1}/{len(model_names)}:[/bold] {model_name}")

            # Apply per-model backend overrides
            effective_backend = model_overrides.get("backend", backend)
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
                backend_override=effective_backend,
                storages=storages,
                attention_backend=effective_attention_backend.upper()
                if effective_attention_backend
                else None,
                task_overrides=task_overrides,
                model_overrides=model_overrides,
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


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def tasks(filter: str) -> None:
    """List all available tasks in the registry."""
    task_names = list_tasks()
    variants = list_variants()
    regimes = list_regimes()

    if not task_names:
        console.print("[dim]No tasks registered.[/dim]")
        return

    table = Table(title="Available Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Variants", style="green")
    table.add_column("Regimes", style="dim")

    for name in task_names:
        if filter.lower() in name.lower():
            task_variants = variants.get(name, [])
            task_regimes = regimes.get(name, [])
            variant_str = ", ".join(task_variants) if task_variants else "-"
            regime_str = ", ".join(task_regimes) if task_regimes else "-"
            table.add_row(name, variant_str, regime_str)

    console.print(table)


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def models(filter: str) -> None:
    """List available model presets."""
    table = Table(title="Model Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="dim")

    for name, cfg in sorted(get_model_presets().items()):
        if filter.lower() in name.lower():
            table.add_row(name, cfg.model)

    console.print(table)


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def suites(filter: str) -> None:
    """List available task suites (task groups)."""
    table = Table(title="Task Suites")
    table.add_column("Suite", style="cyan")
    table.add_column("Tasks", style="dim")
    table.add_column("Aggregation", style="yellow")

    for name in list_suites():
        if filter.lower() in name.lower():
            suite = get_suite(name)
            task_count = len(suite.expanded_tasks)
            table.add_row(name, f"{task_count} tasks", suite.aggregation.value)

    console.print(table)


@main.command(name="suite-info")
@click.argument("suite_name")
def suite_info(suite_name: str) -> None:
    """Show tasks and regimes in a suite.

    SUITE_NAME is the name of the suite to inspect.

    Example: olmo-eval suite-info core
    """
    try:
        suite = get_suite(suite_name)
    except KeyError:
        console.print(f"[red]Error:[/red] Suite '{suite_name}' not found")
        console.print(f"\n[dim]Available suites: {', '.join(list_suites())}[/dim]")
        raise SystemExit(1) from None

    # Header with suite info
    console.print(f"\n[bold cyan]Suite:[/bold cyan] {suite.name}")
    if suite.description:
        console.print(f"[dim]{suite.description}[/dim]")
    console.print(f"[bold]Aggregation:[/bold] {suite.aggregation.value}")
    console.print()

    # Table of tasks
    table = Table(title=f"Tasks in '{suite_name}'")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Task", style="cyan")
    table.add_column("Regime", style="yellow")

    for idx, task_spec in enumerate(suite.expanded_tasks, 1):
        # Parse task::regime format
        if "::" in task_spec:
            task_name, regime = task_spec.split("::", 1)
        else:
            task_name = task_spec
            regime = "(default)"
        table.add_row(str(idx), task_name, regime)

    console.print(table)
    console.print(f"\n[dim]Total: {len(suite.expanded_tasks)} tasks[/dim]")


@beaker.command()
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    help="YAML config file (CLI args override config values)",
)
@click.option("--name", "-n", help="Experiment name")
@click.option(
    "--model",
    "-m",
    multiple=True,
    help="Model name or preset (can specify multiple)",
)
@click.option(
    "--task",
    "-t",
    multiple=True,
    help="Task name with optional @priority suffix (e.g., mmlu, mmlu@high)",
)
@click.option("--cluster", "-c", default=None, help="Cluster alias (h100, a100, aus) or full name")
@click.option("--gpus", "-G", default=None, type=int, help="Number of GPUs per model instance")
@click.option(
    "--parallelism",
    "-P",
    default=None,
    type=int,
    help="Number of model instances to run in parallel",
)
@click.option(
    "--max-gpus-per-node",
    default=None,
    type=int,
    help="Maximum GPUs per node (default: 8). Tasks are split across experiments if exceeded.",
)
@click.option(
    "--priority",
    "-p",
    default=None,
    type=click.Choice(["low", "normal", "high", "urgent"]),
    help="Job priority",
)
@click.option("--preemptible/--no-preemptible", default=None, help="Allow preemption")
@click.option("--timeout", "-T", default=None, help="Job timeout (e.g., 24h, 30m)")
@click.option("--retries", "-r", type=int, help="Number of retries on failure")
@click.option("--workspace", "-w", help="Beaker workspace")
@click.option("--budget", "-B", help="Beaker budget")
@click.option("--image", "-I", help="Beaker image (e.g., ai2-tylerm/olmo-eval-cu1261-trc280-amd64)")
@click.option(
    "--group",
    "-g",
    multiple=True,
    help="Add experiments to Beaker group(s) (can specify multiple, creates if needed)",
)
@click.option(
    "--backends",
    "-b",
    multiple=True,
    help="Backend optional groups to install at runtime (e.g., vllm, hf, litellm)",
)
@click.option("--async", "-a", "use_async", is_flag=True, help="Enable parallel task execution")
@click.option(
    "--async-stream",
    "use_async_stream",
    is_flag=True,
    help="Enable streaming async with vLLM's AsyncLLMEngine for true continuous batching",
)
@click.option("--num-workers", "-W", type=int, help="Number of workers for async mode")
@click.option("--gpus-per-worker", type=int, default=1, help="GPUs per worker for async mode")
@click.option("--dry-run", "-d", is_flag=True, help="Print spec without launching")
@click.option(
    "--follow/--no-follow",
    default=True,
    help="Follow logs after launch (default). Use --no-follow to submit and exit immediately.",
)
@click.option(
    "--aws-credentials/--no-aws-credentials",
    default=None,
    help="Inject AWS credentials for S3 model access. Auto-detected from s3:// model paths.",
)
@click.option(
    "--gcs-credentials/--no-gcs-credentials",
    default=None,
    help="Inject GCS credentials for gs:// model access. Auto-detected from gs:// model paths.",
)
def launch(
    config: str | None,
    name: str | None,
    model: tuple[str, ...],
    task: tuple[str, ...],
    cluster: str | None,
    gpus: int | None,
    parallelism: int | None,
    max_gpus_per_node: int | None,
    priority: str | None,
    preemptible: bool | None,
    timeout: str | None,
    retries: int | None,
    workspace: str | None,
    budget: str | None,
    image: str | None,
    group: tuple[str, ...],
    backends: tuple[str, ...],
    use_async: bool,
    use_async_stream: bool,
    num_workers: int | None,
    gpus_per_worker: int,
    dry_run: bool,
    follow: bool,
    aws_credentials: bool | None,
    gcs_credentials: bool | None,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-py to be installed: pip install 'olmo-eval-internal[beaker]'

    Multiple models and/or tasks with different priorities will create separate experiments.
    Use --config/-f to load settings from a YAML file; CLI arguments override config values.
    Use --group/-g to organize experiments into a Beaker group for result aggregation.
    Use --backends/-b to install inference backends at runtime (e.g., vllm, transformers).

    Examples:

        olmo-eval beaker launch -n "eval-llama3" -m llama3.1-8b -t mmlu

        olmo-eval beaker launch -n "eval-suite" -m llama3.1-8b -t mmlu -t gsm8k -t arc

        olmo-eval beaker launch -n "eval-70b" -m llama3.1-70b -t mmlu --cluster h100 --gpus 4

        # Multiple models (creates separate experiments per model)
        olmo-eval beaker launch -n "eval-compare" -m llama3.1-8b -m olmo-2-7b -t mmlu -t gsm8k

        # Per-task priorities (creates separate experiments per priority level)
        olmo-eval beaker launch -n "eval-mixed" -m llama3.1-8b -t "mmlu@high" -t "gsm8k@normal"

        # Install backends at runtime
        olmo-eval beaker launch -n "eval-vllm" -m llama3.1-8b -t mmlu -b vllm

        # From YAML config file
        olmo-eval beaker launch -f eval_config.yaml

        # Config file with CLI overrides
        olmo-eval beaker launch -f eval_config.yaml --gpus 4 --priority high

        # With grouping for result aggregation
        olmo-eval beaker launch -n "benchmark" --group "benchmark-2024" \\
            -m llama3.1-8b -t mmlu -t gsm8k
    """
    import json as json_module

    try:
        from olmo_eval.launch import (
            BeakerEnvSecret,
            BeakerJobConfig,
            BeakerLauncher,
            LaunchConfig,
            ModelConfig,
            calculate_experiment_splits,
            get_model_short_name,
            parse_model_config,
            validate_priority_configuration,
        )
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    # Track which CLI args were explicitly set (vs using defaults)
    cli_cluster = cluster
    cli_gpus = gpus
    cli_parallelism = parallelism
    cli_priority = priority
    cli_preemptible = preemptible
    cli_timeout = timeout

    # Load config from file if provided
    cfg: LaunchConfig | None = None
    model_configs: list[ModelConfig] = []

    if config:
        try:
            cfg = LaunchConfig.from_yaml(config)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None
        except Exception as e:
            console.print(f"[red]Config error:[/red] {e}")
            raise SystemExit(1) from None

        # Use config values as defaults, CLI args override
        name = name or cfg.name
        task = task if task else tuple(cfg.tasks)
        backends = backends if backends else (tuple(cfg.backends) if cfg.backends else ())
        retries = retries if retries is not None else cfg.retries
        workspace = workspace or cfg.workspace
        budget = budget or cfg.budget

        # Get model configs from file (with per-model resource overrides)
        if not model:
            model_configs = cfg.get_model_configs()
        else:
            # CLI models override config file models
            model_configs = [parse_model_config(m) for m in model]

        # Set defaults from config (will be overridden by per-model or CLI)
        cluster = cluster if cluster is not None else cfg.cluster
        gpus = gpus if gpus is not None else cfg.gpus
        parallelism = parallelism if parallelism is not None else cfg.parallelism
        if max_gpus_per_node is None:
            max_gpus_per_node = cfg.max_gpus_per_node
        priority = priority if priority is not None else cfg.priority
        preemptible = preemptible if preemptible is not None else cfg.preemptible
        timeout = timeout if timeout is not None else cfg.timeout
        use_async = use_async or cfg.use_async
        num_workers = num_workers if num_workers is not None else cfg.num_workers
        gpus_per_worker = gpus_per_worker if gpus_per_worker != 1 else cfg.gpus_per_worker
    else:
        # No config file - use CLI models
        model_configs = [parse_model_config(m) for m in model] if model else []

    # Apply defaults for values not set by config or CLI
    gpus = gpus if gpus is not None else 1
    parallelism = parallelism if parallelism is not None else 1
    if max_gpus_per_node is None:
        max_gpus_per_node = DEFAULT_MAX_GPUS_PER_NODE
    priority = priority or "normal"
    preemptible = preemptible if preemptible is not None else True
    timeout = timeout or "24h"

    # Validate required fields
    if not name:
        console.print("[red]Error:[/red] --name/-n is required (or set 'name' in config)")
        raise SystemExit(1)
    if not model_configs:
        console.print("[red]Error:[/red] --model/-m is required (or set 'models' in config)")
        raise SystemExit(1)
    if not task:
        console.print("[red]Error:[/red] --task/-t is required (or set 'tasks' in config)")
        raise SystemExit(1)
    if not cluster:
        console.print("[red]Error:[/red] --cluster/-c is required (or set 'cluster' in config)")
        raise SystemExit(1)
    if not workspace:
        console.print("[red]Error:[/red] --workspace/-w is required (or set 'workspace' in config)")
        raise SystemExit(1)
    if not budget:
        console.print("[red]Error:[/red] --budget/-B is required (or set 'budget' in config)")
        raise SystemExit(1)

    # Keep suites unexpanded - let the runner expand them so it knows suite names for aggregation
    from olmo_eval.core.configs import expand_tasks, validate_tasks

    original_task_specs = list(task)  # Preserve original suite/task names

    # Group by priority WITHOUT expanding first - this keeps suites as single units
    # so suite aggregation works correctly in the runner
    try:
        tasks_by_priority = validate_priority_configuration(
            tasks=original_task_specs,  # Pass original specs, NOT expanded
            cli_priority=cli_priority,
            default_priority=priority,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Get all specs (without @priority suffix, but with ::overrides)
    all_task_specs = [t for tasks in tasks_by_priority.values() for t in tasks]

    # Expand for validation only - ensure all tasks/suites exist
    expanded_for_validation = expand_tasks(all_task_specs)
    valid_tasks, invalid_tasks = validate_tasks(expanded_for_validation)

    # Track expanded task counts per priority for display purposes
    expanded_counts_by_priority: dict[str, int] = {}
    for priority_level, specs in tasks_by_priority.items():
        expanded_counts_by_priority[priority_level] = len(expand_tasks(specs))

    if invalid_tasks:
        console.print("[red]Error:[/red] The following tasks/suites do not exist:")
        for inv in invalid_tasks:
            console.print(f"  - {inv}")
        console.print("\nUse 'olmo-eval tasks' to see available tasks.")
        console.print("Use 'olmo-eval suites' to see available suites.")
        raise SystemExit(1)

    launcher = BeakerLauncher(workspace=workspace)
    multiple_models = len(model_configs) > 1
    multiple_priorities = len(tasks_by_priority) > 1

    # Auto-detect S3 model paths for AWS credential injection
    from olmo_eval.launch.aws import get_local_aws_credentials, is_s3_path

    s3_models = [m.name_or_path for m in model_configs if is_s3_path(m.name_or_path)]
    inject_aws_credentials = aws_credentials
    if inject_aws_credentials is None:
        # Auto-detect: check if any model path is S3
        inject_aws_credentials = bool(s3_models)

    if inject_aws_credentials:
        local_creds = get_local_aws_credentials()
        beaker_user = launcher.beaker.user_name

        s3_table = Table(title="S3 Access Configuration", show_header=False, box=None)
        s3_table.add_column("Key", style="blue")
        s3_table.add_column("Value")

        if local_creds:
            cred_type = "temporary" if local_creds.session_token else "long-term"
            s3_table.add_row("Credentials", f"[green]found[/green] ({cred_type})")
            s3_table.add_row("Beaker user", beaker_user)
            s3_table.add_row(
                "Beaker secrets",
                f"{beaker_user}_AWS_ACCESS_KEY_ID, {beaker_user}_AWS_SECRET_ACCESS_KEY",
            )
        else:
            s3_table.add_row(
                "Credentials",
                "[yellow]not found[/yellow] - job may fail if S3 access is required",
            )

        console.print()
        console.print(s3_table)
        console.print()

    # Auto-detect GCS model paths for GCS credential injection
    from olmo_eval.launch.gcs import get_local_gcs_credentials, is_gcs_path

    gcs_models = [m.name_or_path for m in model_configs if is_gcs_path(m.name_or_path)]
    inject_gcs_credentials = gcs_credentials
    if inject_gcs_credentials is None:
        # Auto-detect: check if any model path is GCS
        inject_gcs_credentials = bool(gcs_models)

    if inject_gcs_credentials:
        local_gcs_creds = get_local_gcs_credentials()
        beaker_user = launcher.beaker.user_name

        gcs_table = Table(title="GCS Access Configuration", show_header=False, box=None)
        gcs_table.add_column("Key", style="blue")
        gcs_table.add_column("Value")

        if local_gcs_creds:
            gcs_table.add_row("Credentials", "[green]found[/green] (service account)")
            if local_gcs_creds.client_email:
                gcs_table.add_row("Service account", local_gcs_creds.client_email)
            if local_gcs_creds.project_id:
                gcs_table.add_row("Project", local_gcs_creds.project_id)
            gcs_table.add_row("Beaker user", beaker_user)
            gcs_table.add_row("Beaker secret", f"{beaker_user}_GOOGLE_CREDENTIALS")
        else:
            gcs_table.add_row(
                "Credentials",
                "[yellow]not found[/yellow] - job may fail if GCS access is required",
            )

        console.print()
        console.print(gcs_table)
        console.print()

    # Get workspace object for beaker API calls that require it
    workspace_obj = launcher.beaker.workspace.get(workspace) if workspace else None

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Build list of groups from CLI and config
    # Auto-generate one group name if none specified
    from datetime import datetime

    effective_groups: list[str] = list(group)  # CLI groups
    if cfg is not None and cfg.groups:
        # Add config groups (CLI groups take precedence, so they're first)
        for g in cfg.groups:
            if g not in effective_groups:
                effective_groups.append(g)

    # Auto-generate a group if none specified
    if not effective_groups:
        effective_groups = [f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"]

    # Determine the effective image (CLI overrides config, config overrides default)
    from olmo_eval.core.constants.infrastructure import BEAKER_DEFAULT_IMAGE

    if image:
        effective_image = image
    elif cfg and cfg.beaker_image:
        effective_image = cfg.beaker_image
    else:
        effective_image = BEAKER_DEFAULT_IMAGE

    # Check which groups exist and which need to be created
    from beaker.exceptions import BeakerGroupNotFound

    existing_groups: list[str] = []
    missing_groups: list[str] = []

    for grp in effective_groups:
        qualified_name = f"{launcher.beaker.user_name}/{grp}" if "/" not in grp else grp
        try:
            launcher.beaker.group.get(qualified_name)
            existing_groups.append(grp)
        except BeakerGroupNotFound:
            missing_groups.append(grp)

    if dry_run:
        # In dry-run mode, just inform about groups that would be created
        if missing_groups:
            console.print(
                f"[yellow]Note:[/yellow] The following groups would be created: "
                f"{', '.join(missing_groups)}"
            )
    else:
        # In real mode, prompt to create missing groups
        if missing_groups:
            console.print(
                f"\n[yellow]The following groups do not exist:[/yellow] {', '.join(missing_groups)}"
            )
            if not click.confirm("Would you like to create these groups?", default=True):
                console.print("[red]Aborted.[/red] Cannot launch without required groups.")
                raise SystemExit(1)

            # Create the missing groups
            for grp in missing_groups:
                try:
                    beaker_group = launcher.beaker.group.create(
                        name=grp,
                        workspace=workspace_obj,
                    )
                    group_url = launcher.get_group_url(beaker_group)
                    console.print(f"[green]  Created {grp}:[/green] {group_url}")
                except Exception as e:
                    console.print(f"[red]Error:[/red] Failed to create group '{grp}': {e}")
                    raise SystemExit(1) from None

    # Track launched experiments
    launched_experiments: list[str] = []

    # Build experiment plan with parallelism and splitting
    experiment_plan: list[dict] = []
    split_models: list[str] = []  # Track models that require splitting

    for m_cfg in model_configs:
        m_name = m_cfg.name_or_path
        short_m = get_model_short_name(m_cfg)
        if cfg is not None:
            m_resources = cfg.get_model_resources(m_cfg)
            m_gpus = cli_gpus if cli_gpus is not None else m_resources.get("gpus", 1)
            m_parallelism = (
                cli_parallelism
                if cli_parallelism is not None
                else m_resources.get("parallelism", 1)
            )
        else:
            m_gpus = cli_gpus if cli_gpus is not None else (m_cfg.gpus or gpus)
            m_parallelism = (
                cli_parallelism
                if cli_parallelism is not None
                else (m_cfg.parallelism or parallelism)
            )

        for t_priority, t_list in tasks_by_priority.items():
            base_name = name
            if multiple_models:
                base_name = f"{base_name}-{short_m}"
            if multiple_priorities:
                # Use priority level for experiment naming when multiple priorities
                base_name = f"{base_name}-{t_priority}"

            # t_list contains unexpanded specs (suites stay as single units)
            # Splits are based on number of specs, keeping suites together for aggregation
            splits = calculate_experiment_splits(
                tasks=t_list,
                gpus_per_model=m_gpus,
                parallelism=m_parallelism,
                max_gpus_per_node=max_gpus_per_node,
            )

            if len(splits) > 1:
                split_models.append(m_name)

            total_splits = len(splits)
            # Use expanded count for display (actual number of tasks that will run)
            total_expanded = expanded_counts_by_priority[t_priority]
            for i, split in enumerate(splits):
                # Add zero-padded suffix for splits
                exp_name = f"{base_name}-{i + 1:03d}" if total_splits > 1 else base_name

                experiment_plan.append(
                    {
                        "name": exp_name,
                        "model_name": m_name,
                        "model_cfg": m_cfg,
                        "priority": t_priority,
                        "tasks": split["tasks"],
                        "original_task_specs": original_task_specs,  # Original suite/task names
                        "total_expanded_tasks": total_expanded,  # Total tasks in this priority
                        "gpus_per_model": m_gpus,
                        "num_gpus": split["num_gpus"],
                        "parallelism": split["parallelism"],
                        "split_index": i + 1 if total_splits > 1 else None,
                        "total_splits": total_splits if total_splits > 1 else None,
                    }
                )

    # Calculate total expanded tasks
    total_experiments = len(experiment_plan)
    total_expanded_tasks = len(valid_tasks) * len(model_configs)

    # Fetch actual task configurations for display
    from olmo_eval.evals.tasks import get_task as get_task_instance
    from olmo_eval.evals.tasks.core.registry import parse_task_spec

    task_summaries: list[TaskSummary] = []
    for task_spec in valid_tasks:
        # Parse the spec to extract variants/regimes and inline overrides
        task_name, variants, inline_overrides = parse_task_spec(task_spec)

        task_instance = get_task_instance(task_spec)
        task_cfg = task_instance.config
        task_summaries.append(
            TaskSummary(
                name=task_cfg.name,
                spec=task_spec if task_spec != task_cfg.name else None,
                variants=variants if variants else None,
                formatter=task_cfg.formatter,
                scorers=task_cfg.scorers,
                metrics=task_cfg.metrics,
                num_fewshot=task_cfg.num_fewshot,
                split=task_cfg.split.value
                if hasattr(task_cfg.split, "value")
                else str(task_cfg.split),
                primary_metric=str(task_cfg.primary_metric) if task_cfg.primary_metric else None,
                sampling_params=task_cfg.sampling_params,
                overrides=inline_overrides if inline_overrides else None,
            )
        )

    # Build model summaries with resolved backends
    from olmo_eval.core.configs import get_model_config as get_runtime_model_config

    model_summaries: list[ModelSummary] = []
    for m in model_configs:
        # Parse the model spec to extract base name and inline overrides
        model_base_name, model_inline_overrides = parse_model_spec(m.name_or_path)

        # Resolve the effective backend (explicit override or from model preset/default)
        if m.backend:
            effective_backend = m.backend
        else:
            runtime_model_config = get_runtime_model_config(model_base_name)
            effective_backend = runtime_model_config.backend

        model_summaries.append(
            ModelSummary(
                name=model_base_name,
                gpus=m.gpus or gpus,
                parallelism=m.parallelism or parallelism,
                alias=m.alias,
                backend=effective_backend,
                overrides=model_inline_overrides if model_inline_overrides else None,
            )
        )

    # Build runner config
    from olmo_eval.runners import AsyncEvalRunner, StreamingEvalRunner, SyncEvalRunner

    # Resolve effective attention_backend from first model's config if available
    effective_attention_backend = None
    if cfg is not None and model_configs:
        first_model = model_configs[0]
        model_resources = cfg.get_model_resources(first_model)
        effective_attention_backend = model_resources.get("attention_backend")

    if use_async_stream:
        # Resolve effective num_workers from CLI or first model's launch config
        effective_num_workers = num_workers
        if effective_num_workers is None and cfg is not None and model_configs:
            first_model = model_configs[0]
            model_resources = cfg.get_model_resources(first_model)
            effective_num_workers = model_resources.get("num_workers")

        runner_config = RunnerConfig(
            runner=StreamingEvalRunner,
            output_dir=BEAKER_RESULT_DIR,
            attention_backend=effective_attention_backend,
            num_workers=effective_num_workers if effective_num_workers is not None else "auto",
            gpus_per_worker=gpus_per_worker,
        )
    elif use_async:
        # Resolve effective num_workers from CLI or first model's launch config
        effective_num_workers = num_workers
        if effective_num_workers is None and cfg is not None and model_configs:
            first_model = model_configs[0]
            model_resources = cfg.get_model_resources(first_model)
            effective_num_workers = model_resources.get("num_workers")

        runner_config = RunnerConfig(
            runner=AsyncEvalRunner,
            output_dir=BEAKER_RESULT_DIR,
            attention_backend=effective_attention_backend,
            num_workers=effective_num_workers if effective_num_workers is not None else "auto",
            gpus_per_worker=gpus_per_worker,
        )
    else:
        runner_config = RunnerConfig(
            runner=SyncEvalRunner,
            output_dir=BEAKER_RESULT_DIR,
            attention_backend=effective_attention_backend,
        )

    # Build the complete launch summary dataclass
    launch_config_summary = LaunchSummary(
        models=model_summaries,
        tasks=task_summaries,
        runner=runner_config,
    )

    # Print consolidated launch configuration using rich repr
    console.print()
    console.print(
        Panel(
            Pretty(launch_config_summary, expand_all=True),
            title="[bold]Launch Configuration[/bold]",
            border_style="blue",
        )
    )

    # Print experiment summary
    console.print(
        f"\n[bold]Experiments:[/bold] {total_experiments} experiment(s), "
        f"{total_expanded_tasks} task(s)"
    )
    if split_models:
        console.print(
            "[dim]  Tasks distributed across multiple experiments due to GPU constraints[/dim]"
        )

    # Simplified experiment table - only show if multiple experiments or verbose
    if total_experiments > 1:
        matrix_table = Table(show_header=True, title="Experiment Plan")
        matrix_table.add_column("Name", style="cyan")
        matrix_table.add_column("Model", style="blue")
        matrix_table.add_column("Priority", style="yellow")
        matrix_table.add_column("Tasks", justify="right")
        matrix_table.add_column("GPUs", style="green", justify="right")

        for exp in experiment_plan:
            task_count = len(exp["tasks"])
            total_tasks = exp["total_expanded_tasks"]
            task_display = (
                f"{task_count}/{total_tasks}" if exp["split_index"] is not None else str(task_count)
            )
            matrix_table.add_row(
                exp["name"],
                exp["model_name"],
                exp["priority"],
                task_display,
                str(exp["num_gpus"]),
            )

        console.print(matrix_table)

    console.print()

    # Ensure common secrets exist in Beaker
    from olmo_eval.launch.secrets import (
        ensure_common_secrets,
        get_local_hf_token,
        get_local_wandb_api_key,
    )

    beaker_username = launcher.beaker.user_name
    if dry_run:
        # In dry-run mode, just show what secrets would be used
        common_secrets: list[tuple[str, str]] = []
        if get_local_hf_token():
            common_secrets.append(("HF_TOKEN", f"{beaker_username}_HF_TOKEN"))
        if get_local_wandb_api_key():
            common_secrets.append(("WANDB_API_KEY", f"{beaker_username}_WANDB_API_KEY"))
    else:
        # Ensure secrets exist in Beaker (writes them if missing)
        common_secrets = ensure_common_secrets(workspace=workspace)

    # Build all BeakerJobConfig objects first (before confirmation)
    # so we can display them in the summary
    job_configs: list[BeakerJobConfig] = []

    for exp in experiment_plan:
        model_cfg = exp["model_cfg"]
        model_name = exp["model_name"]
        exp_name = exp["name"]
        task_list = exp["tasks"]
        exp_num_gpus = exp["num_gpus"]
        exp_parallelism = exp["parallelism"]
        effective_priority = exp["priority"]

        # Get effective resources for this model (per-model overrides merged with defaults)
        if cfg is not None:
            model_resources = cfg.get_model_resources(model_cfg)
        else:
            # No config file - use ModelConfig values or defaults
            m_para = model_cfg.parallelism
            model_resources = {
                "gpus": model_cfg.gpus if model_cfg.gpus is not None else gpus,
                "parallelism": m_para if m_para is not None else parallelism,
                "cluster": model_cfg.cluster if model_cfg.cluster is not None else cluster,
                "preemptible": (
                    model_cfg.preemptible if model_cfg.preemptible is not None else preemptible
                ),
                "timeout": model_cfg.timeout if model_cfg.timeout is not None else timeout,
                "shared_memory": model_cfg.shared_memory,
                "backend": model_cfg.backend,
            }

        # CLI args always override per-model config
        # Cast values from model_resources dict to expected types
        effective_cluster: str = (
            cli_cluster if cli_cluster is not None else str(model_resources["cluster"])
        )
        effective_preemptible: bool = (
            cli_preemptible if cli_preemptible is not None else bool(model_resources["preemptible"])
        )
        effective_timeout: str = (
            cli_timeout if cli_timeout is not None else str(model_resources["timeout"])
        )
        res_shared_memory = model_resources.get("shared_memory")
        effective_shared_memory: str = str(res_shared_memory) if res_shared_memory else "10GiB"

        # Build model spec with inline overrides for per-model vLLM loading options
        model_spec = model_name
        model_inline_overrides: list[str] = []

        effective_load_format = model_resources.get("load_format")
        if effective_load_format:
            model_inline_overrides.append(f"load_format={effective_load_format}")

        effective_extra_loader_config = model_resources.get("extra_loader_config")
        if effective_extra_loader_config:
            # Serialize extra_loader_config as compact JSON (no spaces) for inline spec
            json_config = json_module.dumps(effective_extra_loader_config, separators=(",", ":"))
            model_inline_overrides.append(f"extra_loader_config={json_config}")

        if model_inline_overrides:
            model_spec = f"{model_name}::{','.join(model_inline_overrides)}"

        # Build command with this model and experiment's tasks
        command = ["olmo-eval", "run", "-m", model_spec]
        for t in task_list:
            command.extend(["-t", t])

        # Add parallelism if > 1 (so the run command knows to run multiple instances)
        if exp_parallelism > 1:
            command.extend(["--parallelism", str(exp_parallelism)])

        # Add async flags if enabled (CLI flags override config)
        effective_use_async = use_async or model_resources.get("use_async", False)
        effective_use_async_stream = use_async_stream or model_resources.get(
            "use_async_stream", False
        )
        effective_num_workers = (
            num_workers if num_workers is not None else model_resources.get("num_workers")
        )
        effective_gpus_per_worker = (
            gpus_per_worker if gpus_per_worker != 1 else model_resources.get("gpus_per_worker", 1)
        )

        # --async-stream takes precedence over --async
        if effective_use_async_stream:
            command.append("--async-stream")
            if effective_num_workers is not None:
                command.extend(["--num-workers", str(effective_num_workers)])
            if effective_gpus_per_worker and effective_gpus_per_worker != 1:
                command.extend(["--gpus-per-worker", str(effective_gpus_per_worker)])
        elif effective_use_async:
            command.append("--async")
            if effective_num_workers is not None:
                command.extend(["--num-workers", str(effective_num_workers)])
            if effective_gpus_per_worker and effective_gpus_per_worker != 1:
                command.extend(["--gpus-per-worker", str(effective_gpus_per_worker)])

        # Determine the backend this model will use at runtime
        # First check for explicit backend override in config, then get from model config
        from olmo_eval.core.configs import get_model_config as get_runtime_model_config
        from olmo_eval.core.constants.infrastructure import BACKEND_OPTIONAL_GROUPS

        config_backend = model_resources.get("backend")  # Explicit override from launch config
        if config_backend:
            runtime_backend: str = str(config_backend)
        else:
            # Get the backend from model config (preset or default)
            runtime_model_config = get_runtime_model_config(model_name)
            runtime_backend = runtime_model_config.backend

        # CLI backends override auto-detected backend optional group
        if backends:
            effective_backends = list(backends)
        else:
            # Get the optional group name for this backend
            backend_group = BACKEND_OPTIONAL_GROUPS.get(runtime_backend)
            effective_backends = [backend_group] if backend_group else []

        # Convert common secrets to BeakerEnvSecret objects
        env_secrets = [
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in common_secrets
        ]

        job_config = BeakerJobConfig(
            name=exp_name,
            command=command,
            cluster=effective_cluster,
            num_gpus=exp_num_gpus,
            priority=effective_priority,
            preemptible=effective_preemptible,
            timeout=effective_timeout,
            shared_memory=effective_shared_memory,
            retries=retries,
            workspace=workspace,
            budget=budget,
            backends=effective_backends,
            groups=effective_groups,
            beaker_image=effective_image,
            inject_aws_credentials=inject_aws_credentials,
            inject_gcs_credentials=inject_gcs_credentials,
            env_secrets=env_secrets,
        )
        job_configs.append(job_config)

    # Display all BeakerJobConfig objects
    for job_config in job_configs:
        console.print(
            Panel(
                Pretty(job_config, expand_all=True),
                title=f"[bold]BeakerJobConfig: {job_config.name}[/bold]",
                border_style="cyan",
            )
        )

    # Confirm before launching (skip in dry-run mode)
    if not dry_run and not click.confirm("Proceed with launch?", default=True):
        console.print("[yellow]Launch cancelled[/yellow]")
        raise SystemExit(0)

    # Launch experiments
    for job_config in job_configs:
        if not dry_run:
            experiment = launcher.launch(job_config)
            if experiment:
                console.print(f"[green]Launched:[/green] {launcher.experiment_url(experiment)}")
                launched_experiments.append(experiment.id)

    # Summary and follow logic for launched experiments
    if launched_experiments and not dry_run:
        # Only show summary count for multiple experiments
        if len(launched_experiments) > 1:
            console.print(f"\n[bold]Launched {len(launched_experiments)} experiment(s)[/bold]")

        # Follow experiment(s) if requested
        if follow:
            if len(launched_experiments) == 1:
                # Single experiment: follow it
                import sys

                exit_code = launcher.follow_experiment(launched_experiments[0])
                sys.exit(exit_code)
            else:
                # Multiple experiments: don't follow, show URLs for watch command
                console.print(
                    "\n[bold]Multiple experiments launched. "
                    "Use 'olmo-eval beaker watch -e <id>' to follow:[/bold]"
                )
                for exp_id in launched_experiments:
                    url = launcher.get_experiment_url(exp_id)
                    console.print(f"  - {url}")


@beaker.command(hidden=True)
@click.option("--group", "-g", required=True, help="Beaker group name")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "csv", "json"]),
    default="table",
    help="Output format",
)
@click.option("--wait", is_flag=True, help="Wait for all experiments to complete")
@click.option(
    "--poll-interval",
    type=int,
    default=30,
    help="Seconds between status checks when waiting",
)
@click.pass_context
def results(
    ctx: click.Context,
    group: str,
    output_format: str,
    wait: bool,
    poll_interval: int,
) -> None:
    """[DEPRECATED] Use 'olmo-eval beaker group info' instead.

    This command is deprecated. Please use:

        olmo-eval beaker group info <group_name> [options]

    Examples:

        olmo-eval beaker group info my-group
        olmo-eval beaker group info my-group --wait --format csv > results.csv
    """
    console.print(
        "[yellow]Warning:[/yellow] 'olmo-eval beaker results' is deprecated.\n"
        f"Use: olmo-eval beaker group info {group}"
        + (" --wait" if wait else "")
        + (f" --format {output_format}" if output_format != "table" else "")
        + "\n"
    )
    # Delegate to group_info
    ctx.invoke(
        group_info,
        group_name=group,
        output_format=output_format,
        verbose=False,
        wait=wait,
        poll_interval=poll_interval,
    )


@beaker.command(name="watch")
@click.option(
    "--experiment",
    "-e",
    required=True,
    help="Beaker experiment ID to watch",
)
@click.option(
    "--tail",
    "-t",
    is_flag=True,
    help="Only show recent logs (last 10 seconds). Useful for attaching to running experiments.",
)
def watch(experiment: str, tail: bool) -> None:
    """Watch an experiment's logs in real-time.

    Streams logs from a Beaker experiment until it completes. Shows startup
    events (pulling image, scheduling) followed by live log output.

    Use --tail/-t to show only recent logs when attaching to an already-running
    experiment.

    Examples:

        # Watch an experiment from the start
        olmo-eval beaker watch -e 01abc123

        # Attach to a running experiment (show recent logs only)
        olmo-eval beaker watch -e 01abc123 --tail
    """
    import sys

    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()

    try:
        exit_code = launcher.follow_experiment(experiment, tail=tail)
        sys.exit(exit_code)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None


@beaker.group()
def group() -> None:
    """Manage Beaker groups.

    Commands for viewing group status, getting detailed task info,
    and bulk operations like canceling all experiments.
    """
    pass


@group.command(name="info")
@click.argument("group_name")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "csv", "json"]),
    default="table",
    help="Output format (csv exports raw metrics from Beaker)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed task info")
@click.option("--wait", "-w", is_flag=True, help="Wait for all experiments to complete")
@click.option(
    "--poll-interval",
    type=int,
    default=30,
    help="Seconds between status checks when waiting",
)
def group_info(
    group_name: str, output_format: str, verbose: bool, wait: bool, poll_interval: int
) -> None:
    """Get detailed info about a Beaker group.

    Shows status of all experiments and tasks in the group.
    Use --wait to block until all experiments complete.

    Examples:

        olmo-eval beaker group info my-experiment-group

        olmo-eval beaker group info my-experiment-group --verbose

        olmo-eval beaker group info my-experiment-group --format json

        # Wait for completion and export as CSV
        olmo-eval beaker group info my-experiment-group --wait --format csv > results.csv
    """
    import json as json_module

    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()

    # Try to get the group
    try:
        from beaker.exceptions import BeakerGroupNotFound

        beaker_group = launcher.beaker.group.get(group_name)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group_name}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Wait for completion if requested
    if wait:
        import time

        console.print(f"[dim]Waiting for experiments in '{group_name}' to complete...[/dim]")
        while True:
            status = launcher.get_group_status(beaker_group)
            running = status.get("running", 0) + status.get("pending", 0)

            if running == 0:
                break

            console.print(
                f"[dim]  {status.get('succeeded', 0)} succeeded, "
                f"{status.get('running', 0)} running, "
                f"{status.get('pending', 0)} pending, "
                f"{status.get('failed', 0)} failed[/dim]"
            )
            time.sleep(poll_interval)

        console.print("[green]All experiments completed.[/green]\n")

    # Get status summary
    status = launcher.get_group_status(beaker_group)
    experiments = launcher.get_group_experiments(beaker_group)
    group_url = launcher.get_group_url(beaker_group)

    if output_format == "csv":
        # Export raw metrics CSV from Beaker
        try:
            csv_data = launcher.export_group_metrics(beaker_group)
            click.echo(csv_data)
        except Exception as e:
            from beaker import BeakerWorkloadStatus

            console.print(f"[yellow]Warning:[/yellow] Could not export metrics: {e}")
            # Fall back to basic experiment info
            click.echo("experiment_id,name,status")
            for exp in experiments:
                workload = launcher.beaker.workload.get(exp.id)
                click.echo(f"{exp.id},{exp.name},{BeakerWorkloadStatus(workload.status).name}")

    elif output_format == "json":
        # Build detailed experiment data
        from beaker import BeakerWorkloadStatus

        exp_data = []
        for exp in experiments:
            workload = launcher.beaker.workload.get(exp.id)
            status_enum = BeakerWorkloadStatus(workload.status)
            exp_info = {
                "id": exp.id,
                "name": exp.name,
                "status": status_enum.name,
                "url": launcher.experiment_url(exp),
            }

            # Add task-level details if verbose
            if verbose:
                try:
                    task_list = []
                    for task in exp.tasks:
                        # Convert status int to BeakerWorkloadStatus enum
                        task_status = (
                            BeakerWorkloadStatus(task.status).name if task.status else "unknown"
                        )
                        task_list.append(
                            {
                                "id": task.id,
                                "name": task.name,
                                "status": task_status,
                            }
                        )
                    exp_info["tasks"] = task_list
                except Exception:
                    pass

            exp_data.append(exp_info)

        data = {
            "group": group_name,
            "group_id": beaker_group.id,
            "url": group_url,
            "status": status,
            "total_experiments": len(experiments),
            "experiments": exp_data,
        }
        click.echo(json_module.dumps(data, indent=2))
    else:
        # Table format
        console.print(f"\n[bold]Group:[/bold] {group_name}")
        console.print(f"[bold]ID:[/bold] {beaker_group.id}")
        console.print(f"[bold]URL:[/bold] {group_url}")
        console.print()

        # Status summary
        total = sum(status.values())
        console.print(
            f"[bold]Status Summary:[/bold] {total} experiment(s)\n"
            f"  [green]✓ {status.get('succeeded', 0)} succeeded[/green]\n"
            f"  [yellow]● {status.get('running', 0)} running[/yellow]\n"
            f"  [dim]○ {status.get('pending', 0)} pending[/dim]\n"
            f"  [red]✗ {status.get('failed', 0)} failed[/red]\n"
            f"  [red]⊘ {status.get('canceled', 0)} canceled[/red]"
        )
        console.print()

        if experiments:
            from beaker import BeakerWorkloadStatus

            table = Table(title="Experiments")
            table.add_column("Name", style="cyan")
            table.add_column("Status")
            if verbose:
                table.add_column("Tasks")
            table.add_column("URL", style="dim")

            for exp in experiments:
                workload = launcher.beaker.workload.get(exp.id)
                status_str = BeakerWorkloadStatus(workload.status).name
                status_style = {
                    "succeeded": "[green]succeeded[/green]",
                    "failed": "[red]failed[/red]",
                    "running": "[yellow]running[/yellow]",
                    "canceled": "[red]canceled[/red]",
                }.get(status_str.lower(), f"[dim]{status_str}[/dim]")

                if verbose:
                    # Get task-level details
                    try:
                        task_info = []
                        for task in exp.tasks:
                            # Convert status int to BeakerWorkloadStatus enum
                            task_status = (
                                BeakerWorkloadStatus(task.status).name if task.status else "unknown"
                            )
                            task_info.append(f"{task.name}: {task_status}")
                        task_str = "\n".join(task_info) if task_info else "-"
                    except Exception:
                        task_str = "-"

                    table.add_row(
                        exp.name,
                        status_style,
                        task_str,
                        launcher.experiment_url(exp),
                    )
                else:
                    table.add_row(
                        exp.name,
                        status_style,
                        launcher.experiment_url(exp),
                    )

            console.print(table)
        else:
            console.print("[dim]No experiments in group.[/dim]")


@group.command(name="cancel")
@click.argument("group_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def group_cancel(group_name: str, yes: bool) -> None:
    """Cancel all active experiments in a Beaker group.

    Stops all running and pending experiments. Completed experiments are skipped.

    Examples:

        olmo-eval beaker group cancel my-experiment-group

        olmo-eval beaker group cancel my-experiment-group --yes
    """
    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()

    # Try to get the group
    try:
        from beaker.exceptions import BeakerGroupNotFound

        beaker_group = launcher.beaker.group.get(group_name)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group_name}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Get current status to show what will be affected
    status = launcher.get_group_status(beaker_group)
    active_count = status.get("running", 0) + status.get("pending", 0)

    if active_count == 0:
        console.print(f"[yellow]No active experiments in group '{group_name}'[/yellow]")
        console.print(
            f"Status: {status.get('succeeded', 0)} succeeded, "
            f"{status.get('failed', 0)} failed, "
            f"{status.get('canceled', 0)} canceled"
        )
        return

    # Confirm cancellation
    console.print(f"[bold]Group:[/bold] {group_name}")
    console.print(
        f"[bold]Active experiments:[/bold] {active_count} "
        f"({status.get('running', 0)} running, {status.get('pending', 0)} pending)"
    )

    if not yes and not click.confirm(f"Cancel all {active_count} active experiment(s)?"):
        console.print("[dim]Cancelled.[/dim]")
        return

    # Perform cancellation
    console.print(f"\n[yellow]Canceling {active_count} experiment(s)...[/yellow]")
    result = launcher.cancel_group(beaker_group)

    # Show results
    console.print(
        f"\n[bold]Results:[/bold]\n"
        f"  [green]✓ {result.get('canceled', 0)} canceled[/green]\n"
        f"  [dim]○ {result.get('skipped', 0)} skipped (already completed)[/dim]"
    )
    if result.get("failed", 0) > 0:
        console.print(f"  [red]✗ {result.get('failed', 0)} failed to cancel[/red]")


@group.command(name="list")
@click.option("--workspace", "-w", required=True, help="Beaker workspace to list groups from")
@click.option("--limit", "-n", type=int, default=20, help="Number of groups to show")
@click.option("--search", "-s", help="Search by name or description")
@click.option("--mine/--all", default=True, help="Show only my groups (default) or all groups")
def group_list(workspace: str, limit: int, search: str | None, mine: bool) -> None:
    """List Beaker groups.

    Shows recent groups with their status summaries. By default, only shows
    groups created by the current user. Use --all to show all groups.

    Examples:

        olmo-eval beaker group list -w ai2/oe-data

        olmo-eval beaker group list -w ai2/oe-data --all

        olmo-eval beaker group list -w ai2/oe-data --search "benchmark" --limit 10
    """
    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()

    # Get workspace object for beaker API calls that require it
    workspace_obj = launcher.beaker.workspace.get(workspace) if workspace else None

    # Get current user ID for filtering
    current_user_id = None
    if mine:
        try:
            current_user_id = launcher.beaker.user.get(launcher.beaker.user_name).id
        except Exception:
            console.print(
                "[yellow]Warning: Could not get current user, showing all groups[/yellow]"
            )

    try:
        # Fetch more than limit if filtering by user, since we filter client-side
        fetch_limit = limit * 5 if mine and current_user_id else limit
        all_groups = list(
            launcher.beaker.group.list(
                workspace=workspace_obj,
                name_or_description=search,
                limit=fetch_limit,
            )
        )

        # Filter to current user's groups if requested
        if mine and current_user_id:
            groups = [g for g in all_groups if g.author_id == current_user_id][:limit]
        else:
            groups = all_groups[:limit]
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    if not groups:
        console.print("[dim]No groups found.[/dim]")
        return

    # Cache workspace lookups
    workspace_names: dict[str, str] = {}

    # Status value mappings (from BeakerWorkloadStatus)
    RUNNING_STATUSES = {1, 2, 3, 4, 5, 6, 10}  # submitted, queued, initializing, running, etc.
    SUCCEEDED_STATUS = 8
    FAILED_STATUS = 9

    table = Table(title="Beaker Groups")
    table.add_column("Name", style="cyan")
    table.add_column("Workspace", style="dim")
    table.add_column("Experiments", justify="right")
    table.add_column("Status")
    table.add_column("Created", style="dim")

    for grp in groups:
        try:
            # Get experiment info from task metrics
            task_metrics = list(launcher.beaker.group.list_task_metrics(grp))

            # Count unique experiments and their statuses
            experiments: dict[str, int] = {}  # exp_id -> worst status
            for tm in task_metrics:
                exp_id = tm.experiment_id
                # Keep the worst status (failed > running > succeeded)
                if exp_id not in experiments:
                    experiments[exp_id] = tm.task_status
                elif tm.task_status == FAILED_STATUS:
                    experiments[exp_id] = FAILED_STATUS
                elif tm.task_status in RUNNING_STATUSES and experiments[exp_id] == SUCCEEDED_STATUS:
                    experiments[exp_id] = tm.task_status

            exp_count = len(experiments)

            if exp_count > 0:
                succeeded = sum(1 for s in experiments.values() if s == SUCCEEDED_STATUS)
                failed = sum(1 for s in experiments.values() if s == FAILED_STATUS)
                running = sum(1 for s in experiments.values() if s in RUNNING_STATUSES)
                status_str = (
                    f"[green]{succeeded}[/green]/[yellow]{running}[/yellow]/[red]{failed}[/red]"
                )
            else:
                status_str = "[dim]empty[/dim]"

            # Format creation time from protobuf Timestamp
            created_str = "-"
            if grp.created and grp.created.seconds:
                from datetime import datetime

                created_dt = datetime.fromtimestamp(grp.created.seconds, tz=UTC)
                created_str = created_dt.strftime("%Y-%m-%d %H:%M")

            # Get workspace name (with caching)
            workspace_name = "-"
            if grp.workspace_id:
                if grp.workspace_id not in workspace_names:
                    try:
                        ws = launcher.beaker.workspace.get(grp.workspace_id)
                        workspace_names[grp.workspace_id] = ws.name
                    except Exception:
                        workspace_names[grp.workspace_id] = grp.workspace_id
                workspace_name = workspace_names[grp.workspace_id]

            table.add_row(
                grp.name,
                workspace_name,
                str(exp_count),
                status_str,
                created_str,
            )
        except Exception:
            table.add_row(grp.name, "-", "?", "[dim]error[/dim]", "-")

    console.print(table)


if __name__ == "__main__":
    main()
