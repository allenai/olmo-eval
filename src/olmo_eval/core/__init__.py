"""Core abstractions for evaluation."""

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
from .logging import (
    configure_logging,
    configure_worker_logging,
    get_logger,
    get_worker_id,
)
from .metrics import (
    AccuracyMetric,
    BPBMetric,
    F1Metric,
    MeanPerplexityMetric,
    Metric,
    PassAtKMetric,
)
from .scorers import (
    BitsPerByteScorer,
    CodeExecutionScorer,
    ExactMatchScorer,
    F1Scorer,
    LogprobScorer,
    MultipleChoiceScorer,
    PerplexityScorer,
    Scorer,
)
from .types import (
    EvalResult,
    Instance,
    LMOutput,
    LMRequest,
    MetricName,
    RequestType,
    Response,
    SamplingParams,
    Split,
    StoredTaskResult,
    compute_model_hash,
    compute_task_hash,
)
from .utils import compute_pass_at_k

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
    "EvalResult",
    "Instance",
    "LMRequest",
    "LMOutput",
    "Response",
    "RequestType",
    "SamplingParams",
    "StoredTaskResult",
    # Utilities
    "compute_model_hash",
    "compute_task_hash",
    # Logging
    "configure_logging",
    "configure_worker_logging",
    "get_logger",
    "get_worker_id",
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
