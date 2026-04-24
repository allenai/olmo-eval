"""Base Task class and configuration."""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmo_eval.common.formatters import Formatter
from olmo_eval.common.metrics import Metric
from olmo_eval.common.repr import hide_unset
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    MetricName,
    RequestType,
    Response,
    SamplingParams,
    Split,
)

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.data import DataSource


@dataclass(frozen=True, slots=True)
class SandboxEnv:
    """Named sandbox environment with dependencies for code execution scoring.

    Tasks sharing the same name share a container. Different names get
    isolated containers with only their declared dependencies installed.
    """

    name: str
    dependencies: tuple[str, ...] = ()
    dockerfile_extra: tuple[str, ...] = ()

    @property
    def capability(self) -> frozenset[str]:
        """Capability tag used to route execution to this sandbox."""
        return frozenset({f"sandbox:{self.name}"})


@hide_unset()
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
    metrics: tuple[Metric, ...] = ()
    num_fewshot: int = 0
    fewshot_seed: int = 42
    limit: int | None = None
    seed: int = 42
    split: Split = Split.TEST
    primary_metric: MetricName | Metric | None = None
    sampling_params: SamplingParams | None = None

    #: Maximum prompt length in tokens for loglikelihood truncation (matches oe-eval max_length).
    #: When set, prompts exceeding this length are left-truncated before scoring.
    max_length: int | None = None

    #: Runtime dependencies to install for this task (package specs like "pkg==1.0" or git URLs)
    dependencies: list[str] | None = None

    #: Sandbox environment for code execution scoring. None = use default sandbox.
    sandbox_env: SandboxEnv | None = None

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
            "metrics": [m.to_dict() for m in self.metrics],
            "num_fewshot": self.num_fewshot,
            "fewshot_seed": self.fewshot_seed,
            "limit": self.limit,
            "seed": self.seed,
            "split": self.split.value,
            "primary_metric": serialize_primary_metric(self.get_primary_metric()),
            "sampling_params": asdict(self.sampling_params) if self.sampling_params else None,
            "max_length": self.max_length,
            "dependencies": self.dependencies,
        }

    def get_primary_metric(self) -> Metric | None:
        """Get the effective primary metric for this task.

        Returns the explicitly set primary_metric if available, otherwise
        returns the single metric if exactly one is defined. Returns None
        if no metrics are defined or multiple metrics exist without an
        explicit primary.

        Returns:
            The primary Metric instance, or None.
        """
        if self.primary_metric is not None:
            # If it's a Metric instance, return it directly
            if isinstance(self.primary_metric, Metric):
                return self.primary_metric
            # If it's a MetricName enum, we can't resolve it to an instance here
            return None
        # Default to single metric if only one is defined
        if len(self.metrics) == 1:
            return self.metrics[0]
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
        self._scorers_cache: dict[str, Scorer] | None = None
        self._has_async_cache: bool | None = None

    @property
    def request_type(self) -> RequestType:
        """The type of request this task produces."""
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.COMPLETION

    def get_sampling_params(self, instance: Instance) -> SamplingParams | None:
        """Get sampling params for a specific instance.

        Override to provide instance-specific sampling params (e.g., per-instance stop sequences).
        Default returns the task-level sampling params.
        """
        return self.config.sampling_params

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

    def extract_answer(self, output: LMOutput) -> Any:
        """Extract the answer from model output."""
        return output.text

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
        return self.config.get_data_source(split=split)

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        """Apply all scorers to extract answers and compute scores.

        Args:
            responses: Responses to score.
            context: Optional scoring context with execution environment for
                scorers that need sandboxed execution.

        Returns:
            The scored responses.

        Subclasses needing custom answer extraction should override
        `_extract_answers()` rather than this method.
        """
        self._extract_answers(responses)

        # Check if any scorers need async execution
        has_async_scorers = self._has_async_scorers()

        if has_async_scorers and context is not None:
            # Run async scoring
            await self._apply_scorers_async(responses, context)
        else:
            # Run sync scoring
            self._apply_scorers(responses)

        return responses

    def _get_scorers(self) -> dict[str, Scorer]:
        """Return scorer instances, keyed by name. Cached after first call."""
        if self._scorers_cache is None:
            scorers_by_name: dict[str, Scorer] = {}
            for metric in self.config.metrics:
                if hasattr(metric, "scorer") and metric.scorer is not None:
                    scorer_instance = metric.scorer()
                    if scorer_instance.name not in scorers_by_name:
                        scorers_by_name[scorer_instance.name] = scorer_instance
            self._scorers_cache = scorers_by_name
        return self._scorers_cache

    def _has_async_scorers(self) -> bool:
        """Check if any configured scorers require async execution.

        This includes ExecutionScorer (sandboxed code) and ContextScorer (LLM judges).
        """
        if self._has_async_cache is None:
            from olmo_eval.common.scorers.execution import ContextScorer, ExecutionScorer

            self._has_async_cache = any(
                isinstance(s, (ExecutionScorer, ContextScorer))
                or getattr(s, "requires_async", False)
                for s in self._get_scorers().values()
            )
        return self._has_async_cache

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract answers from outputs. Override for custom extraction logic."""
        for response in responses:
            for output in response.outputs:
                output.extracted_answer = self.extract_answer(output)

    def _apply_scorers(self, responses: Sequence[Response]) -> None:
        """Run all scorers synchronously and populate response.scores."""
        scorers_by_name = self._get_scorers()

        for response in responses:
            for scorer in scorers_by_name.values():
                scores = [scorer.score(response.instance, o) for o in response.outputs]
                for i, output in enumerate(response.outputs):
                    if output.metadata is None:
                        output.metadata = {}
                    output.metadata[f"score:{scorer.name}"] = scores[i] if i < len(scores) else 0.0
                response.scores[scorer.name] = max(scores) if scores else 0.0

    async def _apply_scorers_async(
        self,
        responses: Sequence[Response],
        context: ScoringContext,
    ) -> None:
        """Run all scorers with context and populate response.scores.

        ExecutionScorer subclasses are scored via ascore() with the execution
        environment. ContextScorer subclasses (e.g., LLM judges) are scored via
        ascore_with_context() with the full scoring context. Regular scorers
        use score() synchronously.

        Args:
            responses: Responses to score.
            context: Scoring context with execution environment and concurrency settings.

        Raises:
            SandboxRequiredError: If an ExecutionScorer is used without a valid
                execution environment in the context.
        """
        from olmo_eval.common.scorers.execution import (
            ContextScorer,
            ExecutionScorer,
            SandboxRequiredError,
        )

        execution_env = context.execution_env if context.has_execution_env else None
        scorers_by_name = self._get_scorers()

        # Route to task-specific sandbox if sandbox_env is configured
        sandbox_cap = self.config.sandbox_env.capability if self.config.sandbox_env else None
        if sandbox_cap and execution_env is not None and hasattr(execution_env, "get_executor"):
            task_executor = execution_env.get_executor(sandbox_cap)  # ty: ignore[call-non-callable]
        else:
            task_executor = execution_env

        # Separate scorers by type: execution (sandbox), context (LLM judge), sync
        execution_scorers: dict[str, ExecutionScorer] = {}
        context_scorers: dict[str, ContextScorer] = {}
        sync_scorers: dict[str, Scorer] = {}
        for name, scorer in scorers_by_name.items():
            if isinstance(scorer, ExecutionScorer):
                if task_executor is None:
                    raise SandboxRequiredError(
                        f"{scorer.__class__.__name__} requires a sandbox. "
                        "Configure sandboxes in HarnessConfig."
                    )
                execution_scorers[name] = scorer
            elif isinstance(scorer, ContextScorer):
                context_scorers[name] = scorer
            else:
                sync_scorers[name] = scorer

        # Apply sync scorers first (fast, no concurrency needed)
        for response in responses:
            for scorer in sync_scorers.values():
                scores = [scorer.score(response.instance, o) for o in response.outputs]
                # Store individual scores in output metadata for pass@k expansion
                for i, output in enumerate(response.outputs):
                    if output.metadata is None:
                        output.metadata = {}
                    output.metadata[f"score:{scorer.name}"] = scores[i] if i < len(scores) else 0.0
                response.scores[scorer.name] = max(scores) if scores else 0.0

        # Apply async scorers (both execution and context) concurrently
        async_scorers_exist = bool(execution_scorers) or bool(context_scorers)
        if async_scorers_exist:
            # Shared semaphore for sandbox execution — scoped by capability, sized
            # to max_concurrency * running_instances. Prevents overloading the pool.
            exec_semaphore: asyncio.Semaphore | contextlib.nullcontext[None] = (
                contextlib.nullcontext()
            )
            if execution_env is not None and hasattr(execution_env, "get_execution_semaphore"):
                from olmo_eval.harness.sandbox.config import Capability

                sem = execution_env.get_execution_semaphore(sandbox_cap or Capability.DEFAULT)  # ty: ignore[call-non-callable]
                if sem is not None:
                    exec_semaphore = sem

            # Per-call semaphore for context scorers (LLM judges) — they don't
            # consume sandbox slots so per-call throttling is appropriate.
            ctx_semaphore = asyncio.Semaphore(context.scoring_concurrency)

            async def score_execution(
                resp_idx: int, scorer: ExecutionScorer, out_idx: int
            ) -> tuple[int, str, int, float]:
                async with exec_semaphore:
                    response = responses[resp_idx]
                    output = response.outputs[out_idx]
                    assert task_executor is not None
                    score = await scorer.ascore(response.instance, output, task_executor)
                    return (resp_idx, scorer.name, out_idx, score)

            async def score_context(
                resp_idx: int, scorer: ContextScorer, out_idx: int
            ) -> tuple[int, str, int, float]:
                async with ctx_semaphore:
                    response = responses[resp_idx]
                    output = response.outputs[out_idx]
                    score = await scorer.ascore_with_context(response.instance, output, context)
                    return (resp_idx, scorer.name, out_idx, score)

            tasks = []
            for resp_idx, response in enumerate(responses):
                # Add execution scorer tasks
                for scorer in execution_scorers.values():
                    for out_idx in range(len(response.outputs)):
                        tasks.append(score_execution(resp_idx, scorer, out_idx))
                # Add context scorer tasks
                for scorer in context_scorers.values():
                    for out_idx in range(len(response.outputs)):
                        tasks.append(score_context(resp_idx, scorer, out_idx))

            # Run all scoring tasks concurrently
            results = await asyncio.gather(*tasks)

            # Store individual scores in output metadata for pass@k expansion
            # Structure: {resp_idx: {scorer_name: {out_idx: score}}}
            scores_by_response: dict[int, dict[str, dict[int, float]]] = {}
            for resp_idx, scorer_name, out_idx, score in results:
                if resp_idx not in scores_by_response:
                    scores_by_response[resp_idx] = {}
                if scorer_name not in scores_by_response[resp_idx]:
                    scores_by_response[resp_idx][scorer_name] = {}
                scores_by_response[resp_idx][scorer_name][out_idx] = score

            # Store individual scores in output metadata and assign max to response
            for resp_idx, scorer_scores in scores_by_response.items():
                response = responses[resp_idx]
                for scorer_name, output_scores in scorer_scores.items():
                    # Store per-output scores in metadata
                    for out_idx, score in output_scores.items():
                        output = response.outputs[out_idx]
                        if output.metadata is None:
                            output.metadata = {}
                        output.metadata[f"score:{scorer_name}"] = score
                    # Response-level score is max for backward compatibility
                    all_scores = list(output_scores.values())
                    response.scores[scorer_name] = max(all_scores) if all_scores else 0.0

    def _expand_multi_output_responses(self, responses: Sequence[Response]) -> list[Response]:
        """Expand multi-output responses into individual responses for pass@k.

        When num_samples > 1, each Response has multiple outputs but scores are
        aggregated to max. For pass@k computation, we need N separate Response
        objects to count individual passing samples. This method expands each
        multi-output Response into N single-output Responses, preserving
        individual scores stored in output.metadata during scoring.

        Args:
            responses: Responses with potentially multiple outputs each.

        Returns:
            Expanded list where each Response has exactly one output.
        """
        expanded: list[Response] = []
        for response in responses:
            if len(response.outputs) <= 1:
                expanded.append(response)
            else:
                for output in response.outputs:
                    new_response = Response(
                        instance=response.instance,
                        request=response.request,
                        outputs=[output],
                        trajectory=response.trajectory,
                    )
                    # Copy scores from output metadata
                    for key, value in (output.metadata or {}).items():
                        if key.startswith("score:"):
                            scorer_name = key[6:]
                            new_response.scores[scorer_name] = value
                    expanded.append(new_response)
        return expanded

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, dict[str, float]]:
        """Compute all metrics from scored responses.

        Returns metrics in nested structure: {metric_name: {scorer_name: score}}.
        This allows multiple scorers to produce the same metric (e.g., accuracy)
        while preserving which scorer produced which value.
        """
        from olmo_eval.common.metrics import PassAtKMetric, PassPowKMetric

        # Expand multi-output responses for pass@k/pass^k metrics only
        expanded_responses: Sequence[Response] | None = None

        result: dict[str, dict[str, float]] = {}
        for metric in self.config.metrics:
            scorer_name = (
                metric.scorer().name if hasattr(metric, "scorer") and metric.scorer else "default"
            )
            if metric.name not in result:
                result[metric.name] = {}

            # Use expanded responses for pass metrics, original for others
            if isinstance(metric, (PassAtKMetric, PassPowKMetric)):
                if expanded_responses is None:
                    expanded_responses = self._expand_multi_output_responses(responses)
                result[metric.name][scorer_name] = metric.compute(expanded_responses)
            else:
                result[metric.name][scorer_name] = metric.compute(responses)
        return result
