"""Shared utilities for the CLI."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console

if TYPE_CHECKING:
    from olmo_eval.evals.external.base import ExternalEval
    from olmo_eval.evals.tasks.common.base import TaskConfig
    from olmo_eval.harness import HarnessConfig
    from olmo_eval.inference.providers.config import ProviderConfig
    from olmo_eval.launch.beaker.launcher import BeakerJobConfig


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


def _coerce_value(value: str) -> Any:
    """Coerce a string value to its appropriate Python type.

    Handles booleans, integers, floats, JSON objects/arrays, and plain strings.
    """
    # Booleans
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # JSON objects and arrays
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # Integers (including negative)
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)

    # Floats (including negative)
    stripped = value.lstrip("-")
    if stripped.replace(".", "", 1).isdigit() and stripped.count(".") == 1:
        return float(value)

    return value


def parse_key_value_args(
    args: tuple[str, ...] | list[str],
    *,
    coerce_types: bool = True,
) -> dict[str, Any]:
    """Parse key=value arguments and JSON dicts into a unified dictionary.

    Args:
        args: Sequence of strings, each either "key=value" or a JSON dict string.
        coerce_types: If True, coerce string values to bools, ints, floats, or JSON.
            If False, keep all values as strings (except JSON dict merges).

    Returns:
        Dictionary with parsed arguments.

    Raises:
        ValueError: If a JSON dict string is invalid.
    """
    result: dict[str, Any] = {}

    for arg in args:
        if arg.startswith("{"):
            # JSON dict format - merge into result
            try:
                parsed = json.loads(arg)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
                result.update(parsed)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}") from e
        elif "=" in arg:
            key, value = arg.split("=", 1)
            result[key] = _coerce_value(value) if coerce_types else value
        # Silently ignore invalid args (no "=" and not JSON)

    return result


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
    harness: "HarnessConfig"
    runner: RunnerConfig
    beaker: "BeakerJobConfig"


@dataclass
class ConfiguredExternalEval:
    """An external eval configured with provider and arguments."""

    name: str
    provider: "ProviderConfig"
    args: dict[str, Any]
    timeout: str
    required_secrets: tuple[str, ...]
    run_command: str
    sandbox_image: str | None = None
    working_dir: str | None = None
    setup_commands: tuple[str, ...] = ()

    @classmethod
    def from_eval(
        cls,
        eval_instance: "ExternalEval",
        provider: "ProviderConfig",
        args: dict[str, Any] | None = None,
    ) -> "ConfiguredExternalEval":
        """Create from an ExternalEval instance."""
        from olmo_eval.evals.external import SandboxedExternalEval

        # Merge defaults with provided args
        merged_args: dict[str, Any] = {}
        for arg_name, (_, default) in eval_instance.arguments.items():
            if default is not None:
                merged_args[arg_name] = default
        if args:
            merged_args.update(args)

        # Format timeout
        timeout_secs = eval_instance.timeout_seconds
        if timeout_secs >= 3600:
            timeout_str = f"{timeout_secs / 3600:.1f}h"
        else:
            timeout_str = f"{timeout_secs:.0f}s"

        # Extract sandbox-specific fields if available
        sandbox_image = None
        working_dir = None
        setup_commands: tuple[str, ...] = ()
        if isinstance(eval_instance, SandboxedExternalEval):
            sandbox_image = eval_instance.sandbox_image
            working_dir = eval_instance.working_dir
            setup_commands = eval_instance.setup_command

        return cls(
            name=eval_instance.name,
            provider=provider,
            args=merged_args,
            sandbox_image=sandbox_image,
            working_dir=working_dir,
            timeout=timeout_str,
            required_secrets=eval_instance.required_secrets,
            setup_commands=setup_commands,
            run_command=eval_instance.run_command,
        )


@dataclass
class ExternalEvalSummary:
    """Per-experiment summary for external eval beaker launch display."""

    name: str
    evals: list[ConfiguredExternalEval]
    beaker: "BeakerJobConfig"


def _get_isolated_vllm_python() -> str | None:
    """Get the path to the isolated vLLM Python interpreter if available."""
    import os

    vllm_python = os.environ.get("VLLM_PYTHON")
    if not vllm_python:
        default_vllm_venv = "/opt/vllm-venv/bin/python"
        if os.path.exists(default_vllm_venv):
            vllm_python = default_vllm_venv

    if vllm_python and os.path.exists(vllm_python):
        return vllm_python
    return None


def _get_vllm_version() -> str | None:
    """Get vLLM version from isolated venv or current environment."""
    import subprocess

    # First check current environment
    try:
        import vllm  # type: ignore[import-not-found]

        return vllm.__version__
    except ImportError:
        pass

    # Check isolated venv
    vllm_python = _get_isolated_vllm_python()
    if vllm_python:
        try:
            result = subprocess.run(
                [vllm_python, "-c", "import vllm; print(vllm.__version__)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return f"{version} (isolated)"
        except (subprocess.TimeoutExpired, OSError):
            pass

    return None


def _get_transformers_version() -> str | None:
    """Get transformers version from current environment or isolated venv."""
    import subprocess

    # First check current environment
    try:
        import transformers

        return transformers.__version__
    except ImportError:
        pass

    # Check isolated venv as fallback
    vllm_python = _get_isolated_vllm_python()
    if vllm_python:
        try:
            result = subprocess.run(
                [vllm_python, "-c", "import transformers; print(transformers.__version__)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return f"{version} (isolated)"
        except (subprocess.TimeoutExpired, OSError):
            pass

    return None


def print_runtime_environment() -> None:
    """Print runtime environment summary for debugging."""
    import os
    import sys

    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold", width=16)
    table.add_column("Value")

    table.add_row("Python", sys.version.split()[0])

    try:
        import torch  # type: ignore[import-not-found]

        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            table.add_row("CUDA version", str(torch.version.cuda))
            table.add_row("cuDNN version", str(torch.backends.cudnn.version()))
            table.add_row("GPU count", str(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                table.add_row(f"  GPU {i}", torch.cuda.get_device_name(i))
    except ImportError:
        table.add_row("PyTorch", "[dim]NOT INSTALLED[/dim]")

    transformers_version = _get_transformers_version()
    if transformers_version:
        table.add_row("Transformers", transformers_version)
    else:
        table.add_row("Transformers", "[dim]NOT INSTALLED[/dim]")

    vllm_version = _get_vllm_version()
    if vllm_version:
        table.add_row("vLLM", vllm_version)
    else:
        table.add_row("vLLM", "[dim]NOT INSTALLED[/dim]")

    # Show VLLM_PYTHON if set
    vllm_python = os.environ.get("VLLM_PYTHON")
    if vllm_python:
        table.add_row("VLLM_PYTHON", vllm_python)

    console.print()
    console.print(Panel(table, title="Runtime Environment", border_style="blue"))
    console.print()
