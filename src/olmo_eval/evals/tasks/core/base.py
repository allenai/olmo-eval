"""Base Task class and configuration."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.core import (
    Formatter,
    Instance,
    LMOutput,
    LMRequest,
    Metric,
    MetricName,
    Response,
    SamplingParams,
    Scorer,
    Split,
)


@dataclass
class TaskConfig:
    """Configuration for a task."""

    name: str
    hf_dataset: str
    hf_subsets: tuple[str, ...] | None = None
    formatter: Formatter | None = None
    scorers: tuple[Scorer, ...] = ()
    metrics: tuple[Metric, ...] = ()
    num_fewshot: int = 0
    fewshot_seed: int = 42
    limit: int | None = None
    split: Split = Split.TEST
    primary_metric: MetricName | None = None
    sampling_params: SamplingParams | None = None
    formatter_overrides: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Abstract base class for evaluation tasks."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self._fewshot_cache: list[Instance] | None = None

    @property
    @abstractmethod
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        ...

    @abstractmethod
    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        ...

    @abstractmethod
    def extract_answer(self, output: LMOutput) -> Any:
        """Extract the answer from model output."""
        ...

    def get_fewshot(self) -> list[Instance]:
        """Get few-shot examples (cached after first call)."""
        if self._fewshot_cache is None:
            self._fewshot_cache = self._build_fewshot()
        return self._fewshot_cache

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples. Override for custom behavior."""
        return []

    def score_responses(self, responses: Sequence[Response]) -> Sequence[Response]:
        """Apply all scorers to extract answers and compute scores."""
        for response in responses:
            for output in response.outputs:
                output.extracted_answer = self.extract_answer(output)
            # Apply each scorer, taking best score across outputs (for multi-sample)
            for scorer in self.config.scorers:
                scores = [scorer.score(response.instance, o) for o in response.outputs]
                response.scores[scorer.name] = max(scores) if scores else 0.0
        return responses

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, float]:
        """Compute all metrics from scored responses."""
        return {m.name: m.compute(responses) for m in self.config.metrics}
