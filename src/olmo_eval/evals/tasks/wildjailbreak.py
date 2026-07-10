"""
WildJailbreak Safety Evaluation Task

This module implements the WildJailbreak evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2406.18510

Usage:

    olmo-eval run -m llama3.1-8b -t wildjailbreak:wg_judge
"""

import logging
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("wildjailbreak")
class WildJailbreak(SafetyBase):
    """wildjailbreak safety evaluation task."""

    data_source = DataSource(path="allenai/olmo-eval-wildjailbreak", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        if doc["data_type"] == "adversarial_benign":
            prompt_type = "benign"
            gold_answer = "response_refusal=no"
        elif doc["data_type"] == "adversarial_harmful":
            prompt_type = "harmful"
            gold_answer = "response_refusal=yes"
        else:
            logger.warning("Failed to parse prompt data type: %s", doc)
            return None

        return Instance(
            question=doc["adversarial"],
            gold_answer=gold_answer,
            metadata={
                "id": index,
                "prompt_type": prompt_type,
            },
        )


_WILDJAILBREAK_SUBSET_METRICS = (
    "prompt_type::benign",
    "prompt_type::harmful",
)

# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("wildjailbreak", _WILDJAILBREAK_SUBSET_METRICS)
