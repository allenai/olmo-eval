"""SimpleQA evaluation task.

This module implements the SimpleQA factual question answering evaluation task.
It can be run with or without search tools via the Harness abstraction.

Usage:
    # With LLM-as-judge scoring
    olmo-eval run -m llama3.1-8b -t simpleqa:judge

    # With search tools (agent evaluation)
    olmo-eval run -m llama3.1-8b -t simpleqa:judge --harness dr_tulu
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import SimpleQAJudgeScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, TaskConfig, register, register_variant

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers factual questions accurately.

When answering questions:
1. If you have access to search tools and are unsure about a fact, use them to find
   accurate information.
2. Provide concise, accurate answers based on the information you have or find.
3. If you cannot find reliable information, say so rather than guessing.

Always strive to give factually correct answers."""


@register("simpleqa")
class SimpleQA(Task):
    """SimpleQA factual question answering evaluation task."""

    data_source = DataSource(path="allenai/simpleqa_full", split="test")
    formatter = ChatFormatter()
    sampling_params = SamplingParams(temperature=0.0)

    default_source: str = "allenai/simpleqa_full"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            try:
                source = self.config.get_data_source()
            except ValueError:
                source = DataSource(path=self.default_source, split="test")

            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)

        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # Handle different possible field names
        # The dataset may have question directly, or in messages format
        question = doc.get("question") or doc.get("problem") or ""

        # Handle messages format: [{"role": "user", "content": "..."}]
        if not question and "messages" in doc:
            messages = doc["messages"]
            if messages and len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg, dict) and first_msg.get("role") == "user":
                    question = first_msg.get("content", "")

        gold_answer = doc.get("answer") or doc.get("ground_truth") or doc.get("gold_answer") or ""

        if not question:
            return None

        return Instance(
            question=question,
            gold_answer=gold_answer,
            metadata={
                "id": doc.get("id", f"simpleqa_{index}"),
                "index": index,
                "dataset": "simpleqa",
            },
        )

    @property
    def request_type(self) -> RequestType:
        """Return the request type for this task."""
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        Delegates to the configured formatter (ChatFormatter by default).
        """
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        # Fallback: create a simple chat request
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )


# =============================================================================
# Variant Registrations
# =============================================================================

# Judge variant - uses LLM-as-judge scoring for factual accuracy
register_variant(
    "simpleqa",
    "judge",
    metrics=(AccuracyMetric(scorer=SimpleQAJudgeScorer),),
)
