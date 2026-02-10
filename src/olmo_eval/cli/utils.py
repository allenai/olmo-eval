"""Shared utilities for the CLI."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console

if TYPE_CHECKING:
    from olmo_eval.evals.tasks.common.base import TaskConfig
    from olmo_eval.harness import HarnessConfig
    from olmo_eval.launch.beaker.launcher import BeakerJobConfig


@dataclass
class HarnessSummary:
    """Display-friendly representation of a Harness."""

    config: "HarnessConfig"


console = Console(force_terminal=True, width=120)


@dataclass
class FlaggedArg:
    """Argument with its flag type for order tracking."""

    flag: str  # 't', 'o', or 'h'
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
    -t, -o, and --harness options appeared on the command line.

    Args:
        args: List of command line arguments (e.g., sys.argv[1:]).

    Returns:
        List of FlaggedArg in the order they appeared.
    """
    # Map option flags to their short flag character
    flag_map = {
        "-t": "t",
        "--task": "t",
        "-o": "o",
        "--override": "o",
        "-H": "h",
        "--harness": "h",
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
) -> tuple[dict[str, list[str]], list[str]]:
    """Associate -o overrides with preceding -t or --harness.

    Args:
        ordered: List of FlaggedArg with flag type and value.

    Returns:
        Tuple of (task_overrides, harness_overrides) where:
        - task_overrides is a dict mapping task name to list of override strings
        - harness_overrides is a list of override strings for the harness

    Raises:
        click.UsageError: If -o appears without a preceding -t or --harness.
    """
    task_overrides: dict[str, list[str]] = {}
    harness_overrides: list[str] = []

    current_task: str | None = None
    last_flag: str | None = None

    for arg in ordered:
        if arg.flag == "t":
            # Strip priority suffix (@urgent, @high, etc.) for override key
            current_task = arg.value.rsplit("@", 1)[0] if "@" in arg.value else arg.value
            task_overrides.setdefault(current_task, [])
            last_flag = "t"
        elif arg.flag == "h":
            last_flag = "h"
        elif arg.flag == "o":
            # Apply to task or harness
            if last_flag == "t" and current_task:
                task_overrides[current_task].append(arg.value)
            elif last_flag == "h":
                harness_overrides.append(arg.value)
            else:
                raise click.UsageError("-o/--override must follow -t/--task or --harness")

    return task_overrides, harness_overrides


def extract_priority_from_overrides(
    task_overrides: dict[str, list[str]],
) -> tuple[str | None, dict[str, list[str]]]:
    """Extract priority from task overrides and return filtered overrides.

    If any task has a 'priority=X' override, it's used to set the job priority.
    The priority override is removed from the returned task overrides (it's not
    a valid task config field).

    Args:
        task_overrides: Dict of task_spec -> override strings.

    Returns:
        Tuple of (extracted_priority, filtered_task_overrides).
    """
    extracted_priority: str | None = None
    filtered: dict[str, list[str]] = {}

    for task_spec, overrides in task_overrides.items():
        new_overrides = []
        for override in overrides:
            if override.startswith("priority="):
                # Extract priority value
                extracted_priority = override.split("=", 1)[1]
            else:
                new_overrides.append(override)
        if new_overrides:
            filtered[task_spec] = new_overrides

    return extracted_priority, filtered


def format_timestamp(ts: datetime | None) -> str:
    """Format a timestamp for display."""
    if ts is None:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class RunnerConfig:
    """Runner configuration for display."""

    runner: type
    output_dir: str | None = None
    attention_backend: str | None = None

    def __repr__(self) -> str:
        parts = [f"runner={self.runner.__name__}"]
        if self.output_dir is not None:
            parts.append(f"output_dir={self.output_dir!r}")
        if self.attention_backend is not None:
            parts.append(f"attention_backend={self.attention_backend!r}")
        return f"RunnerConfig({', '.join(parts)})"


@dataclass
class ExperimentSummary:
    """Per-experiment summary for beaker launch display."""

    name: str
    tasks: list["TaskConfig"]
    harness: HarnessSummary
    runner: RunnerConfig
    beaker: "BeakerJobConfig"


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
