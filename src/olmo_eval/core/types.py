"""Core data types and enums for evaluation."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, ClassVar


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


class Split(str, Enum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class MetricName(str, Enum):
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


@dataclass(frozen=True, slots=True)
class Instance:
    """A single evaluation instance."""

    question: str
    gold_answer: str | None = None
    choices: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LMRequest:
    """Request to send to a language model.

    For CHAT requests: use `messages`
    For COMPLETION requests: use `prompt` and optionally `continuations`
    """

    request_type: RequestType
    # Chat-style fields
    messages: tuple[dict[str, str], ...] = ()
    # Completion-style fields
    prompt: str = ""
    continuations: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters for language model sampling."""

    #: Fields that can be overridden via inline task specs (e.g., task::temperature=0.5)
    OVERRIDE_KEYS: ClassVar[set[str]] = {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "num_samples",
    }

    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    num_samples: int = 1
    logprobs: int | None = None


@dataclass(slots=True)
class LMOutput:
    """Output from a language model."""

    text: str
    logprobs: list[dict[str, Any]] | None = None
    extracted_answer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Response:
    """Complete response pairing instance, request, and outputs."""

    instance: Instance
    request: LMRequest
    outputs: list[LMOutput] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class StoredTaskResult:
    """Result for a single task within an evaluation.

    Stores task-level metrics and references to storage locations where
    detailed predictions and metrics files are stored.
    """

    task_name: str
    metrics: dict[str, float]
    task_hash: str
    task_config: dict[str, Any] | None = None
    num_instances: int | None = None
    primary_metric: str | None = None
    primary_score: float | None = None
    # Storage references for detailed data
    s3_metrics_key: str | None = None
    s3_predictions_key: str | None = None
    s3_requests_key: str | None = None


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

    def __post_init__(self) -> None:
        """Compute model_hash from model_config if not provided."""
        if self.model_hash is None and self.model_config is not None:
            self.model_hash = compute_model_hash(self.model_config)
