"""
Do_Anything_Now Safety Evaluation Task

This module implements the Do_Anything_Now evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2308.03825

Usage:

    olmo-eval run -m llama3.1-8b -t do_anything_now:wg_judge
"""

import logging
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import ChatFormatter, CompletionFormatter
from olmo_eval.common.metrics import AccuracyMetric, SafetyErrorMetric, SubsetAccuracyMetric
from olmo_eval.common.scorers import SafetyScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_think_answer, extract_think_answer_only
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)


@register("do_anything_now")
class DoAnythingNow(Task):
    """DoAnythingNow safety evaluation task."""

    data_source = DataSource(path="allenai/tulu-3-do-anything-now-eval", split="test")
    formatter = ChatFormatter()
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95)
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

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""

        return Instance(
            question=doc.get("adversarial"),
            metadata={
                "id": index,
                "jailbreak_source": doc.get("source"),
                "platform": doc.get("platform"),
                "gold_answer": "response_refusal=yes",
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
            return self.config.formatter.format(instance)
        # Fallback: create a simple chat request
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )


_SAFETY_SUBSET_METRICS = (
    "jailbreak_source::jailbreak_chat",
    "jailbreak_source::BreakGPT",
    "jailbreak_source::ChatGPT",
    "jailbreak_source::LLM Promptwriting",
    "jailbreak_source::AI Prompt Sharing",
    "jailbreak_source::ChatGPTJailbreak",
)

_JUDGE_SAMPLING = SamplingParams(max_tokens=32768, temperature=0.7, top_p=0.95)


def _safety_metrics(scorer):
    """Build the full metric tuple for a safety judge scorer."""
    return (
        AccuracyMetric(scorer=scorer),
        SafetyErrorMetric(scorer=scorer),
        *(SubsetAccuracyMetric(name=name, scorer=scorer) for name in _SAFETY_SUBSET_METRICS),
    )


# =============================================================================
# Variant Registrations
# =============================================================================

# OpenAI judge variant - uses OpenAI API as the judge
register_variant(
    "do_anything_now",
    "openai_judge",
    metrics=_safety_metrics(SafetyScorer),
    primary_metric=AccuracyMetric(scorer=SafetyScorer),
    sampling_params=_JUDGE_SAMPLING,
)

# Wildguard judge variant - uses a local auxiliary provider (auxiliary_providers.wg_judge)
_WG_SCORER = SafetyScorer(
    provider_name="wg_judge",
    judge_format="wildguard",
    judge_request_type=RequestType.COMPLETION,
)

register_variant(
    "do_anything_now",
    "wg_judge",
    metrics=_safety_metrics(_WG_SCORER),
    primary_metric=AccuracyMetric(scorer=_WG_SCORER),
    sampling_params=_JUDGE_SAMPLING,
)

register_variant(
    "do_anything_now",
    "wg_judge_thinking",
    metrics=_safety_metrics(_WG_SCORER),
    primary_metric=AccuracyMetric(scorer=_WG_SCORER),
    sampling_params=_JUDGE_SAMPLING,
    answer_extractor=extract_think_answer_only,
)

register_variant(
    "do_anything_now",
    "base",
    metrics=_safety_metrics(_WG_SCORER),
    primary_metric=AccuracyMetric(scorer=_WG_SCORER),
    sampling_params=_JUDGE_SAMPLING,
    formatter=CompletionFormatter(),
    judge_request_type=RequestType.COMPLETION,
)
