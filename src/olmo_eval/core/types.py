"""Core data types and enums for evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, ClassVar


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


@dataclass(frozen=True, slots=True)
class Result:
    """Result of a completed evaluation task."""

    experiment_id: str
    experiment_name: str
    workspace: str
    created: str
    author_name: str
    tags: str
    git_ref: str
    model_hash: str
    model_name: str
    revision: str
    regimes: str
    task_hash: str
    task_name: str
    primary_metric: str
    primary_score: str


@dataclass
class StoredTaskResult:
    """Result for a single task within an evaluation.

    Stores task-level metrics and references to storage locations where
    detailed predictions and metrics files are stored.
    """

    task_name: str
    metrics: dict[str, float]
    num_instances: int | None = None
    task_hash: str | None = None
    primary_metric: str | None = None
    primary_score: float | None = None
    # Storage references for detailed data
    s3_metrics_key: str | None = None
    s3_predictions_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "task_name": self.task_name,
            "metrics": self.metrics,
        }
        if self.num_instances is not None:
            result["num_instances"] = self.num_instances
        if self.task_hash is not None:
            result["task_hash"] = self.task_hash
        if self.primary_metric is not None:
            result["primary_metric"] = self.primary_metric
        if self.primary_score is not None:
            result["primary_score"] = self.primary_score
        if self.s3_metrics_key is not None:
            result["s3_metrics_key"] = self.s3_metrics_key
        if self.s3_predictions_key is not None:
            result["s3_predictions_key"] = self.s3_predictions_key
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredTaskResult":
        """Create from dictionary."""
        return cls(
            task_name=data["task_name"],
            metrics=data["metrics"],
            num_instances=data.get("num_instances"),
            task_hash=data.get("task_hash"),
            primary_metric=data.get("primary_metric"),
            primary_score=data.get("primary_score"),
            s3_metrics_key=data.get("s3_metrics_key"),
            s3_predictions_key=data.get("s3_predictions_key"),
        )


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
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "backend_name": self.backend_name,
            "timestamp": self.timestamp.isoformat(),
            "tasks": [t.to_dict() for t in self.tasks],
        }
        if self.experiment_name is not None:
            result["experiment_name"] = self.experiment_name
        if self.workspace is not None:
            result["workspace"] = self.workspace
        if self.author is not None:
            result["author"] = self.author
        if self.tags is not None:
            result["tags"] = self.tags
        if self.git_ref is not None:
            result["git_ref"] = self.git_ref
        if self.model_hash is not None:
            result["model_hash"] = self.model_hash
        if self.revision is not None:
            result["revision"] = self.revision
        if self.s3_location is not None:
            result["s3_location"] = self.s3_location
        if self.config is not None:
            result["config"] = self.config
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            model_name=data["model_name"],
            backend_name=data["backend_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tasks=[StoredTaskResult.from_dict(t) for t in data.get("tasks", [])],
            experiment_name=data.get("experiment_name"),
            workspace=data.get("workspace"),
            author=data.get("author"),
            tags=data.get("tags"),
            git_ref=data.get("git_ref"),
            model_hash=data.get("model_hash"),
            revision=data.get("revision"),
            s3_location=data.get("s3_location"),
            config=data.get("config"),
            metadata=data.get("metadata"),
        )
