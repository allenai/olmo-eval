"""Core data types and enums for evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum, auto
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

if TYPE_CHECKING:
    from .tools import ToolCall, ToolSchema
    from .trajectory import AgentTrajectory


def compute_model_hash(config: dict[str, Any] | None) -> str | None:
    """Compute a deterministic hash from a model configuration dict.

    The model hash is used to identify unique model configurations
    across experiments. The same config always produces the same hash,
    allowing multiple experiments (from different users or runs) to be
    associated with the same model configuration.

    Args:
        config: Model configuration dictionary.

    Returns:
        16-character hex string hash of the config, or None if config is None.

    Example:
        >>> config = {"model": "llama3.1-8b", "temperature": 0.7}
        >>> model_hash = compute_model_hash(config)
        >>> len(model_hash)
        16
    """
    if not config:
        return None

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_task_hash(config: dict[str, Any] | None) -> str | None:
    """Compute a deterministic hash from a task configuration dict.

    Args:
        config: Task configuration dictionary.

    Returns:
        16-character hex string hash of the config, or None if config is None.
    """
    if not config:
        return None

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class Split(StrEnum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class MetricName(StrEnum):
    """Standard metric identifiers."""

    ACCURACY = "accuracy"
    ACC_PER_CHAR = "acc_per_char"
    ACC_PER_TOKEN = "acc_per_token"
    EXACT_MATCH = "exact_match"
    PASS_AT_1 = "pass_at_1"
    PASS_AT_K = "pass_at_k"
    F1 = "f1"


class RequestType(Enum):
    """Type of request to send to the LM."""

    CHAT = auto()
    COMPLETION = auto()
    LOGLIKELIHOOD = auto()


class TopLogProb(TypedDict):
    """A single top logprob alternative."""

    token: str
    logprob: float
    bytes: NotRequired[list[int]]


class LogProbEntry(TypedDict):
    """A single logprob entry for a token.

    Compatible with OpenAI's ChatCompletionTokenLogprob format.
    The bytes and top_logprobs fields are optional for backward compatibility.
    """

    token: str
    logprob: float
    bytes: NotRequired[list[int]]
    top_logprobs: NotRequired[list[TopLogProb]]


@dataclass(frozen=True, slots=True)
class Instance:
    """A single evaluation instance.

    The base fields (question, gold_answer, choices, metadata) support
    traditional evaluation. The tool-related fields support agent and
    tool calling evaluation.
    """

    question: str
    gold_answer: str | None = None
    choices: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tools: tuple[ToolSchema, ...] | None = None
    expected_tool_calls: tuple[dict[str, Any], ...] | None = None
    should_abstain: bool | None = None
    required_trajectory: tuple[dict[str, Any], ...] | None = None
    initial_state: dict[str, Any] | None = None
    expected_final_state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LMRequest:
    """Request to send to a language model."""

    request_type: RequestType
    messages: tuple[dict[str, Any], ...] = ()
    prompt: str = ""
    continuations: tuple[str, ...] | None = None
    tools: tuple[ToolSchema, ...] | None = None
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters for language model sampling."""

    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    num_samples: int = 1
    logprobs: int | None = None


@dataclass(slots=True)
class LMOutput:
    """Output from a language model.

    Supports both text generation and tool calling outputs.
    """

    text: str
    logprobs: list[LogProbEntry] | None = None
    extracted_answer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if this output contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


@dataclass(slots=True)
class Response:
    """Complete response pairing instance, request, and outputs.

    For multi-turn agent evaluations, the trajectory field contains
    the complete interaction history.
    """

    instance: Instance
    request: LMRequest
    outputs: list[LMOutput] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    trajectory: AgentTrajectory | None = None


@dataclass
class StoredTaskResult:
    """Result for a single task within an evaluation.

    Stores task-level metrics and references to storage locations where
    detailed predictions and metrics files are stored.

    Metrics are stored in a nested structure:
        metrics = {
            "metric_name": {
                "scorer_name": score_value,
            },
        }
    For example:
        metrics = {
            "accuracy": {"exact_match": 0.85, "simpleqa_judge": 0.72},
            "not_attempted_rate": {"simpleqa_judge": 0.15},
        }

    The primary_metric field uses "metric_name:scorer_name" format to identify
    the primary score for display purposes.
    """

    task_name: str
    metrics: dict[str, dict[str, float]]
    task_hash: str
    task_config: dict[str, Any] | None = None
    num_instances: int | None = None
    primary_metric: str | None = None  # Format: "metric_name:scorer_name"
    # Storage references for detailed data
    s3_metrics_key: str | None = None
    s3_predictions_key: str | None = None
    s3_requests_key: str | None = None
    # Duration tracking
    duration_seconds: float | None = None


@dataclass
class EvalResult:
    """Complete result for an evaluation run.

    Stores run-level metadata and references to storage locations where
    the full evaluation data (completions, metrics, predictions) is stored.

    Fields align with the evaluation tracking schema:
    - Core identifiers: experiment_id, model_name, backend_name
    - Experiment info: experiment_name, workspace, author, tags
    - Version tracking: git_ref, model_hash, revision
    - Storage reference: s3_location points to base path with all task results

    Note: experiment_id can be shared across multiple models in a single
    experiment launch.
    """

    experiment_id: str
    model_name: str
    backend_name: str
    timestamp: datetime
    tasks: list[StoredTaskResult] = field(default_factory=list)
    # Experiment metadata
    experiment_name: str | None = None
    workspace: str | None = None
    author: str | None = None
    tags: list[str] | None = None
    # Version tracking
    git_ref: str | None = None
    model_hash: str | None = None
    revision: str | None = None
    # Storage reference - base path where all task results are stored
    s3_location: str | None = None
    # Flexible config and metadata
    model_config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    # Original model path (when alias is used, model_name is the alias)
    model_path: str | None = None
    # Experiment group for grouping related experiments
    experiment_group: str | None = None
    # Duration metrics
    experiment_duration_seconds: float | None = None
    provider_init_seconds: dict[str, float] | None = None  # model_name -> init_time

    def __post_init__(self) -> None:
        """Compute model_hash from model_config if not provided."""
        if self.model_hash is None and self.model_config is not None:
            self.model_hash = compute_model_hash(self.model_config)
