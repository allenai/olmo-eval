"""Core data types and enums for evaluation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


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
