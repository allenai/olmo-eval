"""
HarmBench Safety Evaluation Task

This module implements the HarmBench Safety evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2402.04249

Usage:

    olmo-eval run -m llama3.1-8b -t harmbench:wg_judge
"""

import logging
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, SafetyErrorMetric, SubsetAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataLoader
from olmo_eval.evals.extract import extract_think_answer
from olmo_eval.evals.tasks.common import Task

logger = logging.getLogger(__name__)

judge_sampling = SamplingParams(max_tokens=32768, temperature=0.7, top_p=0.95)
base_sampling = SamplingParams(
    max_tokens=1024,
    temperature=0.6,
    top_p=0.6,
    stop_sequences=("Question:", "</s>", "<|im_end|>", "\n\n"),
)


class SafetyBase(Task):
    """Base class for safety evaluation tasks."""

    data_source = None
    formatter = ChatFormatter()
    sampling_params = judge_sampling
    answer_extractor = extract_think_answer

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()

            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)

        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """
        Convert a dataset document to an Instance. This method must be implemented
        by each task individually.
        """
        return None

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
            return self.config.formatter.format(instance)
        # Fallback: create a simple chat request
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )


def safety_metrics(scorer, subsets):
    """Build the full metric tuple for a safety judge scorer."""
    return (
        AccuracyMetric(scorer=scorer),
        SafetyErrorMetric(scorer=scorer),
        *(SubsetAccuracyMetric(name=name, scorer=scorer) for name in subsets),
    )
