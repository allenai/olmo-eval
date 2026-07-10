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

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

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

register_safety_variants("do_anything_now", _DOANYTHING_SUBSET_METRICS)
