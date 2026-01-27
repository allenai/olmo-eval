"""Shared utilities for the CLI."""

from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()

# Keys that apply to model/backend config
MODEL_KEYS = {
    "backend",
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
    backend: str | None = None
    overrides: dict[str, Any] | None = None


@dataclass
class TaskSummary:
    """Summary of a task configuration for display."""

    name: str
    spec: str | None = None
    variants: list[str] | None = None
    formatter: Any = None
    scorers: tuple = ()
    metrics: tuple = ()
    num_fewshot: int = 0
    split: str = "test"
    primary_metric: str | None = None
    sampling_params: Any = None
    overrides: dict[str, Any] | None = None


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
class EvalSummary:
    """Complete launch configuration summary for pretty-printing."""

    models: list[ModelSummary]
    tasks: list[TaskSummary]
    runner: RunnerConfig


def parse_model_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse model spec into (model_name, overrides).

    Format: model[::key=value,...]
    """
    from olmo_eval.evals.tasks.core.registry import parse_overrides

    main_part, _, override_str = spec.partition("::")
    overrides = parse_overrides(override_str) if override_str else {}
    return main_part, overrides


def parse_task_spec_with_overrides(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse task spec with inline overrides.

    Format: task[:variant...][::key=value,...]
    """
    from olmo_eval.evals.tasks.core.registry import parse_overrides

    spec_part, _, override_str = spec.partition("::")
    overrides = parse_overrides(override_str) if override_str else {}
    return spec_part, overrides


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
