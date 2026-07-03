"""
Do_Anything_Now Safety Evaluation Task

This module implements the Do_Anything_Now evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2308.03825

Usage:

    olmo-eval run -m llama3.1-8b -t do_anything_now:wg_judge
"""

import logging
from typing import Any

from olmo_eval.common.formatters import CompletionFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import SafetyScorer
from olmo_eval.common.types import Instance, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import extract_think_answer_only
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.safety_base import (
    SafetyBase,
    base_sampling,
    judge_sampling,
    safety_metrics,
)

logger = logging.getLogger(__name__)


@register("do_anything_now")
class DoAnythingNow(SafetyBase):
    """DoAnythingNow safety evaluation task."""

    data_source = DataSource(path="allenai/tulu-3-do-anything-now-eval", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        return Instance(
            question=doc["adversarial"],
            gold_answer="response_refusal=yes",
            metadata={
                "id": index,
                "jailbreak_source": doc["source"],
                "platform": doc["platform"],
            },
        )


_DOANYTHING_SUBSET_METRICS = (
    "jailbreak_source::jailbreak_chat",
    "jailbreak_source::BreakGPT",
    "jailbreak_source::ChatGPT",
    "jailbreak_source::LLM Promptwriting",
    "jailbreak_source::AI Prompt Sharing",
    "jailbreak_source::ChatGPTJailbreak",
)


# =============================================================================
# Variant Registrations
# =============================================================================

# OpenAI judge variant - uses OpenAI API as the judge
register_variant(
    "do_anything_now",
    "openai_judge",
    metrics=safety_metrics(SafetyScorer, _DOANYTHING_SUBSET_METRICS),
    primary_metric=AccuracyMetric(scorer=SafetyScorer),
    sampling_params=judge_sampling,
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
    metrics=safety_metrics(_WG_SCORER, _DOANYTHING_SUBSET_METRICS),
    primary_metric=AccuracyMetric(scorer=_WG_SCORER),
    sampling_params=judge_sampling,
)

register_variant(
    "do_anything_now",
    "wg_judge_thinking",
    metrics=safety_metrics(_WG_SCORER, _DOANYTHING_SUBSET_METRICS),
    primary_metric=AccuracyMetric(scorer=_WG_SCORER),
    sampling_params=judge_sampling,
    answer_extractor=extract_think_answer_only,
)

register_variant(
    "do_anything_now",
    "base",
    metrics=safety_metrics(_WG_SCORER, _DOANYTHING_SUBSET_METRICS),
    primary_metric=AccuracyMetric(scorer=_WG_SCORER),
    sampling_params=base_sampling,
    formatter=CompletionFormatter(template="Question: {question}\nAnswer:"),
)
