"""
Toxigen Safety Evaluation Task

This module implements the toxigen evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2203.09509

Usage:

    olmo-eval run -m llama3.1-8b -t toxigen:wg_judge
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


@register("toxigen")
class Toxigen(Task):
    """toxigen safety evaluation task."""

    data_source = DataSource(path="allenai/olmo-eval-toxigen", split="test")
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

        if not doc.get("subsample"):
            return None

        return Instance(
            question=doc.get("instruction"),
            gold_answer="response_harmfulness=no",  # used for openai varient not original judge
            metadata={
                "id": doc.get("id"),
                "index": index,
                "target_groups": list(doc.get("target_groups"))[0],
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


_SAFETY_SUBSET_METRICS = (  ##TODO
    "target_groups::asian",
    "target_groups::black",
    "target_groups::chinese",
    "target_groups::jewish",
    "target_groups::latino",
    "target_groups::lgbtq",
    "target_groups::mental_disability",
    "target_groups::mexican",
    "target_groups::middle_east",
    "target_groups::muslim",
    "target_groups::native_american",
    "target_groups::physical_disability",
    "target_groups::trans",
    "target_groups::women",
)

_JUDGE_SAMPLING = SamplingParams(max_tokens=32768, temperature=0.6, top_p=0.95)


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


# tomh/toxigen judge variant - uses a local auxiliary provider (auxiliary_providers.tox_judge)

_TOX_SCORER = SafetyScorer(
    provider_name="wg_judge",
    judge_format="wildguard",
    judge_request_type=RequestType.COMPLETION,
)

register_variant(
    "toxigen",
    "tox_judge",
    metrics=_safety_metrics(_TOX_SCORER),
    primary_metric=AccuracyMetric(scorer=_TOX_SCORER),
    sampling_params=_JUDGE_SAMPLING,
)

register_variant(
    "toxigen",
    "tox_judge_thinking",
    metrics=_safety_metrics(_TOX_SCORER),
    primary_metric=AccuracyMetric(scorer=_TOX_SCORER),
    sampling_params=_JUDGE_SAMPLING,
    answer_extractor=extract_think_answer_only,
)

register_variant(
    "toxigen",
    "base",
    metrics=_safety_metrics(_TOX_SCORER),
    primary_metric=AccuracyMetric(scorer=_TOX_SCORER),
    sampling_params=_JUDGE_SAMPLING,
    formatter=CompletionFormatter(),
    judge_request_type=RequestType.COMPLETION,
)
