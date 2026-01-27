"""Base Task class and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

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

    #: Fields that can be overridden via inline task specs (e.g., task::num_fewshot=5)
    OVERRIDE_KEYS: ClassVar[set[str]] = {"num_fewshot", "limit", "fewshot_seed", "sampling_params"}

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary for hashing and storage.

        Returns:
            Dictionary with all config values serialized.
        """
        from dataclasses import asdict

        def serialize_data_source(ds: Any) -> Any:
            if ds is None:
                return None
            if isinstance(ds, str):
                return ds
            return ds.to_dict()

        def serialize_primary_metric(pm: Any) -> Any:
            if pm is None:
                return None
            # MetricName is a str Enum
            if hasattr(pm, "value"):
                return pm.value
            # Metric instance
            if hasattr(pm, "to_dict"):
                return pm.to_dict()
            return str(pm)

        return {
            "name": self.name,
            "data_source": serialize_data_source(self.data_source),
            "fewshot_source": serialize_data_source(self.fewshot_source),
            "formatter": self.formatter.to_dict() if self.formatter else None,
            "scorers": [s.to_dict() for s in self.scorers],
            "metrics": [m.to_dict() for m in self.metrics],
            "num_fewshot": self.num_fewshot,
            "fewshot_seed": self.fewshot_seed,
            "limit": self.limit,
            "split": self.split.value,
            "primary_metric": serialize_primary_metric(self.primary_metric),
            "sampling_params": asdict(self.sampling_params) if self.sampling_params else None,
        }


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

    # Default data source path (can be overridden by subclasses)
    default_source: str | None = None

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

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a raw document to an Instance.

        Override this method to define how documents are converted to instances.
        Return None to skip the document.

        This is used by `_load_instances()` when using the unified data loader.

        Args:
            doc: A raw document dictionary from the dataset.
            index: The index of the document in the dataset.

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

        for index, doc in enumerate(loader.load(source)):
            instance = self.process_doc(doc, index)
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

    # Class attributes for fewshot configuration (can be overridden by subclasses)
    fewshot_split: str = "dev"
    fewshot_sample: bool = True

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples. Override fewshot_split/fewshot_sample for custom behavior."""
        return self._build_fewshot_from_source(
            split=self.fewshot_split,
            sample=self.fewshot_sample,
        )

    def _build_fewshot_from_source(
        self,
        split: str = "dev",
        sample: bool = True,
        fallback_splits: list[str] | None = None,
    ) -> list[Instance]:
        """Build few-shot examples using the unified data loader.

        Args:
            split: Primary split to use for few-shot examples.
            sample: If True, randomly sample num_fewshot examples. If False, return all.
            fallback_splits: Optional list of splits to try if primary split fails/empty.

        Returns:
            List of Instance objects for few-shot prompting.
        """
        import random

        from olmo_eval.data import DataLoader

        if sample and self.config.num_fewshot == 0:
            return []

        splits_to_try = [split] + (fallback_splits or [])
        all_instances: list[Instance] = []

        loader = DataLoader()
        for try_split in splits_to_try:
            try:
                source = self._get_source_for_split(try_split)
                all_instances = [
                    inst
                    for doc in loader.load(source)
                    if (inst := self.process_doc(doc)) is not None
                ]
                if all_instances:
                    break
            except Exception:
                continue

        if not all_instances:
            return []

        if sample and self.config.num_fewshot:
            rng = random.Random(self.config.fewshot_seed)
            return rng.sample(all_instances, min(self.config.num_fewshot, len(all_instances)))

        return all_instances

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        from olmo_eval.data import DataSource

        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            if self.default_source is not None:
                return DataSource(path=self.default_source, split=split)
            raise

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
