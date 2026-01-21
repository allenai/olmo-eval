"""Core abstractions for evaluation."""

from .code_execution import CodeExecutionScorer, PassAtKMetric, compute_pass_at_k
from .configs import (
    ModelConfig,
    RunConfig,
    expand_tasks,
    get_model_config,
    load_config,
)
from .constants.models import get_model_presets
from .formatters import (
    ChatFormatter,
    CompletionFormatter,
    Formatter,
    MCQAChatFormatter,
    MultipleChoiceFormatter,
    PPLFormatter,
)
from .metrics import AccuracyMetric, BPBMetric, F1Metric, MeanPerplexityMetric, Metric
from .scorers import (
    BitsPerByteScorer,
    ExactMatchScorer,
    F1Scorer,
    LogprobScorer,
    MultipleChoiceScorer,
    PerplexityScorer,
    Scorer,
)
from .types import (
    Instance,
    LMOutput,
    LMRequest,
    MetricName,
    RequestType,
    Response,
    Result,
    SamplingParams,
    Split,
)

__all__ = [
    # Enums
    "Split",
    "MetricName",
    # Configs
    "ModelConfig",
    "RunConfig",
    "get_model_presets",
    "load_config",
    "expand_tasks",
    "get_model_config",
    # Datatypes
    "Instance",
    "LMRequest",
    "LMOutput",
    "Response",
    "Result",
    "RequestType",
    "SamplingParams",
    # Formatters
    "Formatter",
    "ChatFormatter",
    "CompletionFormatter",
    "MCQAChatFormatter",
    "MultipleChoiceFormatter",
    "PPLFormatter",
    # Scoring
    "Scorer",
    "Metric",
    "ExactMatchScorer",
    "MultipleChoiceScorer",
    "F1Scorer",
    "BitsPerByteScorer",
    "PerplexityScorer",
    "LogprobScorer",
    "AccuracyMetric",
    "F1Metric",
    "BPBMetric",
    "MeanPerplexityMetric",
    # Code execution
    "CodeExecutionScorer",
    "PassAtKMetric",
    "compute_pass_at_k",
]
