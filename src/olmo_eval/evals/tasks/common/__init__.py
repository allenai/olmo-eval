"""Core task framework - base classes, registry, and configuration."""

from .base import OutputScoreAggregation, SandboxEnv, Task, TaskConfig
from .cap_f1_judge import DEFAULT_JUDGE_MODEL, CapF1Judge, CapF1Score
from .caption_metrics import (
    CaptionMetricsReport,
    bleu_4,
    compute_caption_metrics,
    rouge_l,
    token_f1,
    tokenize,
)
from .multimodal_base import MultimodalGenerationTask
from .format_helpers import format_mc, format_rc
from .registry import (
    clear_registry,
    get_base_task_name,
    get_sandbox_envs,
    get_task,
    get_task_dependencies,
    list_tasks,
    list_variants,
    parse_overrides,
    parse_task_spec,
    register,
    register_subtasks,
    register_variant,
    task_exists,
)

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "CapF1Judge",
    "CapF1Score",
    "CaptionMetricsReport",
    "MultimodalGenerationTask",
    "OutputScoreAggregation",
    "SandboxEnv",
    "Task",
    "TaskConfig",
    "bleu_4",
    "compute_caption_metrics",
    "rouge_l",
    "token_f1",
    "tokenize",
    "clear_registry",
    "format_mc",
    "format_rc",
    "get_base_task_name",
    "get_sandbox_envs",
    "get_task",
    "get_task_dependencies",
    "list_tasks",
    "list_variants",
    "parse_overrides",
    "parse_task_spec",
    "register",
    "register_subtasks",
    "register_variant",
    "task_exists",
]
