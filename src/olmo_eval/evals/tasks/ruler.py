"""RULER: What's the Real Context Size of Your Long-Context Language Models?

This task implements the RULER benchmark for evaluating long-context language models.
RULER generates synthetic examples to evaluate models across 4 task categories:
- NIAH (Needle in a Haystack): Single/multi-key/multi-value/multi-query variants
- Multi-hop tracing: Variable tracking (VT)
- Aggregation: Common word extraction (CWE), Frequency word extraction (FWE)
- Question Answering: QA with long context

Paper: https://arxiv.org/abs/2404.06654
Original implementation: https://github.com/hsiehjackson/RULER
"""

import os
from collections.abc import Iterator
from typing import Any

from olmo_eval.core.formatters import PPLFormatter
from olmo_eval.core.metrics import AccuracyMetric, BPBMetric, F1Metric, RecallMetric
from olmo_eval.core.scorers import ExactMatchScorer, F1Scorer
from olmo_eval.core.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data.ruler_loader import download_ruler_data, load_ruler_dataset
from olmo_eval.data.ruler_tasks import RULER_TASKS
from olmo_eval.evals.tasks.core.base import Task, TaskConfig
from olmo_eval.evals.tasks.core.registry import register, register_variant


class RulerTask(Task):
    """Base RULER task implementation.

    Each RULER task variant (e.g., niah_s_1__4096) is registered as a separate task
    with specific configuration from RULER_TASKS.
    """

    def __init__(self, config: TaskConfig, task_name: str, ruler_config: dict[str, Any]) -> None:
        """Initialize RULER task.

        Args:
            config: Task configuration
            task_name: Full RULER task name (e.g., "niah_s_1__4096")
            ruler_config: RULER-specific configuration dict
        """
        super().__init__(config)
        self.task_name = task_name
        self.ruler_config = ruler_config

        # Extract context size from task name
        task_type, context_size_str = task_name.rsplit("__", 1)
        self.task_type = task_type
        self.context_size = int(context_size_str)

        # Load dataset during initialization
        self._dataset = None
        self._templates = None

    def _load_data(self) -> None:
        """Load RULER dataset if not already loaded."""
        if self._dataset is not None:
            return

        # Download RULER data if needed
        root_dir = download_ruler_data()

        # Get data path from config
        data_path = os.path.join(root_dir, self.ruler_config["data"])

        # Load dataset
        loaded = load_ruler_dataset(
            task_name=self.task_name,
            data_path=data_path,
            max_samples=self.config.limit,
            seed=42,
        )

        self._dataset = loaded["data"]
        self._templates = {
            "prompt": loaded["prompt_template"],
            "user": loaded["user_template"],
            "system": loaded["system_template"],
        }

    @property
    def instances(self) -> Iterator[Instance]:
        """Generate Instance objects from the dataset."""
        # Load data if needed
        self._load_data()

        # Check cache first
        if self._instances_cache is not None:
            yield from self._instances_cache
            return

        # Generate instances
        self._instances_cache = []
        for idx, doc in enumerate(self._dataset):  # type: ignore
            instance = self.process_doc(doc, index=idx)
            if instance is not None:
                self._instances_cache.append(instance)
                yield instance

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a raw document to an Instance.

        Args:
            doc: Raw document from dataset
            index: Index of the document

        Returns:
            Instance object or None if document should be skipped
        """
        # Build the context by formatting the user template
        context_fields = dict(doc)
        if "context" not in context_fields:
            context_fields["context"] = ""

        # Format the question using the user template
        if self._templates is None:
            raise RuntimeError("Templates not loaded. Call _load_data() first.")
        question = self._templates["user"].format(**context_fields)

        # Add system template as prepend text for non-chat format
        prepend_text = ""
        if not self.ruler_config.get("use_chat_template", False):
            prepend_text = self._templates["system"].format(**context_fields)

        # Get answer (handle both "answer" and "outputs" fields)
        answer = doc.get("answer") or doc.get("outputs")

        return Instance(
            question=question,
            gold_answer=answer,
            metadata={
                "id": doc.get("index", index),
                "task_type": self.task_type,
                "context_size": self.context_size,
                "prepend_text": prepend_text,
                "tag": self.ruler_config["tag"],
            },
        )

    @property
    def request_type(self) -> RequestType:
        """Return the request type for this task."""
        return RequestType.COMPLETION

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance as an LMRequest.

        Args:
            instance: Instance to format

        Returns:
            LMRequest for model inference
        """
        # Use config formatter if provided
        if self.config.formatter is not None:
            # For PPL formatters (BPB variant), convert list answers to strings
            if isinstance(self.config.formatter, PPLFormatter) and isinstance(
                instance.gold_answer, list
            ):
                # Create a modified instance with string answer
                instance = Instance(
                    question=instance.question,
                    gold_answer=", ".join(str(a) for a in instance.gold_answer),
                    metadata=instance.metadata,
                )
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default formatting: just return the prompt
        # (sampling_params are already configured in TaskConfig)
        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Extract answer from model output.

        For RULER tasks, we don't parse the output - we use the raw text
        for substring matching in the scorer.

        Args:
            output: Model output

        Returns:
            Raw output text
        """
        return output.text


# Dynamically register all RULER tasks
for _task_name, _task_config in RULER_TASKS.items():

    def _make_config_factory(task_name: str = _task_name, task_cfg: dict = _task_config):
        """Create config factory for this RULER task variant."""

        def config_factory() -> TaskConfig:
            # Determine metrics based on task type
            if task_cfg["tag"] == "qa":
                # QA tasks use multiple metrics
                metrics = (
                    AccuracyMetric(scorer=ExactMatchScorer),
                    F1Metric(scorer=F1Scorer),
                )
                primary_metric = "exact_match"
            else:
                # Other tasks use substring recall
                metrics = (RecallMetric(),)
                primary_metric = "recall"

            # Build sampling params
            stop_sequences = []
            if task_cfg.get("stop_new_line", False):
                stop_sequences = ["\n", "Ċ", "ĊĊ", "<0x0A>"]

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=task_cfg.get("max_gen_toks", 50),
                stop_sequences=stop_sequences if len(stop_sequences) > 0 else None,
            )

            return TaskConfig(
                name=f"ruler_{task_name}",
                data_source=None,  # Data loading handled internally
                metrics=metrics,
                primary_metric=primary_metric,
                sampling_params=sampling_params,
                limit=100,  # Default limit, can be overridden
            )

        return config_factory

    def _make_task_class(task_name: str = _task_name, task_cfg: dict = _task_config):
        """Create task class for this RULER task variant."""

        class _RulerTask(RulerTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, task_name, task_cfg)

        _RulerTask.__name__ = f"Ruler_{task_name}"
        _RulerTask.__qualname__ = f"Ruler_{task_name}"
        return _RulerTask

    # Register the task
    register(f"ruler_{_task_name}", _make_config_factory())(_make_task_class())

# Register BPB variant for all RULER tasks
# This allows perplexity-based evaluation as an alternative to generation-based
for _task_name in RULER_TASKS:
    register_variant(
        f"ruler_{_task_name}",
        "bpb",
        formatter=PPLFormatter(leading_space=False, answer_prefix=""),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )
