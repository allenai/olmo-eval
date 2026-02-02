"""Shared utilities for the CLI."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console

if TYPE_CHECKING:
    from olmo_eval.launch.beaker.launcher import BeakerJobConfig

console = Console()


@dataclass
class FlaggedArg:
    """Argument with its flag type for order tracking."""

    flag: str  # 'm', 't', or 'o'
    value: str


class OrderedMultiOption(click.Option):
    """Option that tracks order across multiple option types.

    This is a marker class - the actual order tracking is done by
    reconstruct_ordered_args() which parses the original command line.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.save_to: str = kwargs.pop("save_to", "_ordered_args")
        super().__init__(*args, **kwargs)


def reconstruct_ordered_args(args: list[str]) -> list[FlaggedArg]:
    """Reconstruct ordered args from command line arguments.

    Parses the argument list to determine the order in which
    -m, -t, and -o options appeared on the command line.

    Args:
        args: List of command line arguments (e.g., sys.argv[1:]).

    Returns:
        List of FlaggedArg in the order they appeared.
    """
    # Map option flags to their short flag character
    flag_map = {
        "-m": "m",
        "--model": "m",
        "-t": "t",
        "--task": "t",
        "-o": "o",
        "--override": "o",
        "-O": "o",
    }

    ordered: list[FlaggedArg] = []
    i = 0
    while i < len(args):
        arg = args[i]

        # Handle -m=value syntax
        if "=" in arg:
            opt, _, value = arg.partition("=")
            if opt in flag_map:
                ordered.append(FlaggedArg(flag_map[opt], value))
            i += 1
        # Handle -m value syntax
        elif arg in flag_map:
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                ordered.append(FlaggedArg(flag_map[arg], args[i + 1]))
                i += 2
            else:
                i += 1
        else:
            i += 1

    return ordered


def process_ordered_args(
    ordered: list[FlaggedArg],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Associate -o overrides with preceding -m or -t.

    Args:
        ordered: List of FlaggedArg with flag type and value.

    Returns:
        Tuple of (model_overrides, task_overrides) where each is a dict
        mapping model/task name to list of override strings.

    Raises:
        click.UsageError: If -o appears without a preceding -m or -t.
    """
    model_overrides: dict[str, list[str]] = {}  # model_name -> [override_strs]
    task_overrides: dict[str, list[str]] = {}

    current_model: str | None = None
    current_task: str | None = None
    last_flag: str | None = None

    for arg in ordered:
        if arg.flag == "m":
            current_model = arg.value
            model_overrides.setdefault(current_model, [])
            last_flag = "m"
        elif arg.flag == "t":
            current_task = arg.value
            task_overrides.setdefault(current_task, [])
            last_flag = "t"
        elif arg.flag == "o":
            # Apply to last model or task
            if last_flag == "m" and current_model:
                model_overrides[current_model].append(arg.value)
            elif last_flag == "t" and current_task:
                task_overrides[current_task].append(arg.value)
            else:
                raise click.UsageError("-o/--override must follow -m/--model or -t/--task")

    return model_overrides, task_overrides


def format_timestamp(ts: datetime | None) -> str:
    """Format a timestamp for display."""
    if ts is None:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# Keys that apply to model/provider config
MODEL_KEYS = {
    "provider",
    "attention_backend",
    "gpus_per_worker",
    "tokenizer",
    "max_model_len",
    "load_format",
}


@dataclass
class ModelSummary:
    """Summary of a model configuration."""

    name: str
    gpus: int = 1
    parallelism: int = 1
    alias: str | None = None
    provider: str | None = None
    overrides: dict[str, Any] | None = None


@dataclass
class TaskSummary:
    """Summary of a task configuration for display.

    Holds the task config directly to avoid duplicating fields.
    """

    config: Any  # TaskConfig or AgentTaskConfig
    spec: str | None = None
    variants: list[str] | None = None
    overrides: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tool_names(self) -> list[str] | None:
        """Return tool names if this is an agent task with tools."""
        if hasattr(self.config, "tools") and self.config.tools:
            return [t.name for t in self.config.tools]
        return None


@dataclass
class RunnerConfig:
    """Runner configuration for display."""

    runner: type
    output_dir: str | None = None
    attention_backend: str | None = None
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
class ExperimentSummary:
    """Per-experiment summary for beaker launch display."""

    name: str
    models: list[ModelSummary]
    tasks: list[TaskSummary]
    runner: RunnerConfig
    beaker: "BeakerJobConfig"


def parse_model_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse model spec into (model_name, overrides).

    Returns the model name and an empty overrides dict.
    Use -o flag for overrides instead.
    """
    return spec, {}


def parse_task_spec_with_overrides(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse task spec into (task_spec, overrides).

    Returns the task spec and an empty overrides dict.
    Use -o flag for overrides instead.
    """
    return spec, {}


def print_runtime_environment() -> None:
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
