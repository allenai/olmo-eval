"""Base Task class and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from olmo_eval.data import DataSource


@dataclass
class TaskConfig:
    """Configuration for a task.

    Examples:
        # With DataSource object
        >>> from olmo_eval.data import DataSource
        >>> config = TaskConfig(
        ...     name="arc_challenge",
        ...     data_source=DataSource(path="allenai/ai2_arc", subset="ARC-Challenge"),
        ... )

        # With URI string
        >>> config = TaskConfig(
        ...     name="mmlu_math",
        ...     data_source="hf://cais/mmlu?subset=abstract_algebra",
        ... )
    """

    name: str

    # Data source configuration
    data_source: DataSource | str | None = None
    fewshot_source: DataSource | str | None = None

    # Task configuration
    formatter: Formatter | None = None
    scorers: tuple[Scorer, ...] = ()
    metrics: tuple[Metric, ...] = ()
    num_fewshot: int = 0
    fewshot_seed: int = 42
    limit: int | None = None
    split: Split = Split.TEST
    primary_metric: MetricName | Metric | None = None
    sampling_params: SamplingParams | None = None

    def get_data_source(self, split: str | None = None) -> DataSource:
        """Get the data source for a specific split.

        Args:
            split: The split to use. If None, uses the config's default split.

        Returns:
            A DataSource configured for the specified split.

        Raises:
            ValueError: If no data source is configured.
        """
        from olmo_eval.data import DataSource

        if split is None:
            split = self.split.value

        if isinstance(self.data_source, str):
            return DataSource.from_uri(self.data_source, split=split)
        elif isinstance(self.data_source, DataSource):
            return self.data_source.with_split(split)
        raise ValueError("No data source configured for this task")

    def get_fewshot_source(self, split: str = "dev") -> DataSource | None:
        """Get the data source for few-shot examples.

        Args:
            split: The split to use for few-shot examples (default: "dev").

        Returns:
            A DataSource for few-shot examples, or None if not configured.
        """
        from olmo_eval.data import DataSource

        if self.fewshot_source is not None:
            if isinstance(self.fewshot_source, str):
                return DataSource.from_uri(self.fewshot_source, split=split)
            return self.fewshot_source.with_split(split)

        # Fall back to main data source with different split
        try:
            return self.get_data_source(split=split)
        except ValueError:
            return None


class Task(ABC):
    """Abstract base class for evaluation tasks.

    Tasks can either:
    1. Override `instances` property directly (legacy approach)
    2. Implement `process_doc()` and use the default `_load_instances()` helper

    The second approach allows tasks to benefit from the unified data loading
    infrastructure that supports HuggingFace, local files, S3, and GCS sources.

    Example using process_doc:
        >>> class MyTask(Task):
        ...     def process_doc(self, doc: dict) -> Instance:
        ...         return Instance(
        ...             question=doc["question"],
        ...             gold_answer=doc["answer"],
        ...         )
        ...
        ...     @property
        ...     def instances(self) -> Iterator[Instance]:
        ...         yield from self._load_instances()
    """

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self._fewshot_cache: list[Instance] | None = None
        self._instances_cache: list[Instance] | None = None

    @property
    @abstractmethod
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset.

        Subclasses must implement this. They can either:
        1. Implement custom loading logic directly
        2. Use the helper: `yield from self._load_instances()`
        """
        ...

    @abstractmethod
    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        ...

    @abstractmethod
    def extract_answer(self, output: LMOutput) -> Any:
        """Extract the answer from model output."""
        ...

    def process_doc(self, doc: dict[str, Any]) -> Instance | None:
        """Convert a raw document to an Instance.

        Override this method to define how documents are converted to instances.
        Return None to skip the document.

        This is used by `_load_instances()` when using the unified data loader.

        Args:
            doc: A raw document dictionary from the dataset.

        Returns:
            An Instance, or None to skip this document.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process_doc() "
            "to use the unified data loading infrastructure"
        )

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Load and process instances from the configured data source.

        This helper method uses the unified DataLoader to fetch documents
        and calls `process_doc()` to convert them to instances.

        Subclasses can use this in their `instances` property:
            @property
            def instances(self) -> Iterator[Instance]:
                yield from self._load_instances()

        Args:
            split: Optional split override. If None, uses config.split.

        Yields:
            Instance objects from the dataset.
        """
        from olmo_eval.data import DataLoader

        loader = DataLoader()
        source = self.config.get_data_source(split=split)

        for doc in loader.load(source):
            instance = self.process_doc(doc)
            if instance is not None:
                yield instance

    def _load_instances_cached(self, split: str | None = None) -> Iterator[Instance]:
        """Load instances with caching.

        Same as `_load_instances()` but caches results after first call.

        Args:
            split: Optional split override.

        Yields:
            Instance objects from the dataset.
        """
        if self._instances_cache is None:
            self._instances_cache = list(self._load_instances(split=split))
        yield from self._instances_cache

    def get_fewshot(self) -> list[Instance]:
        """Get few-shot examples (cached after first call)."""
        if self._fewshot_cache is None:
            self._fewshot_cache = self._build_fewshot()
        return self._fewshot_cache

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples. Override for custom behavior."""
        return []

    def _build_fewshot_from_source(self, split: str = "dev") -> list[Instance]:
        """Build few-shot examples using the unified data loader.

        Helper method that loads few-shot examples from the configured
        fewshot_source or falls back to the main data source with a
        different split.

        Args:
            split: The split to use for few-shot examples.

        Returns:
            List of Instance objects for few-shot prompting.
        """
        from olmo_eval.data import DataLoader

        source = self.config.get_fewshot_source(split=split)
        if source is None:
            return []

        loader = DataLoader()
        instances = []
        for doc in loader.load(source):
            instance = self.process_doc(doc)
            if instance is not None:
                instances.append(instance)
        return instances

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
