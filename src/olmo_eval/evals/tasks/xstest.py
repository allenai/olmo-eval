"""
XSTest Safety Evaluation Task

This module implements the XSTest evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2308.01263

Usage:

    olmo-eval run -m llama3.1-8b -t xstest:wg_judge
"""

import logging
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("xstest")
class XSTest(SafetyBase):
    """xstest safety evaluation task."""

    data_source = DataSource(path="walledai/XSTest", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""
        prompt_type = doc["label"]

        if prompt_type == "safe":
            gold_answer = "response_refusal=no"
        elif prompt_type == "unsafe":
            gold_answer = "response_refusal=yes"

        return Instance(
            question=doc["prompt"],
            gold_answer=gold_answer,
            metadata={
                "id": index,
                "prompt_type": prompt_type,
                "subcategory": doc["type"],
            },
        )


_XSTEST_SUBSET_METRICS = (
    "prompt_type::safe",
    "prompt_type::unsafe",
    "subcategory::homonyms",
    "subcategory::contrast_homonyms",
    "subcategory::figurative_language",
    "subcategory::contrast_figurative_language",
    "subcategory::safe_targets",
    "subcategory::contrast_safe_targets",
    "subcategory::safe_contexts",
    "subcategory::contrast_safe_contexts",
    "subcategory::definitions",
    "subcategory::contrast_definitions",
    "subcategory::nons_group_real_discr",
    "subcategory::real_group_nons_discr",
    "subcategory::contrast_discr",
    "subcategory::historical_events",
    "subcategory::contrast_historical_events",
    "subcategory::privacy_public",
    "subcategory::privacy_fictional",
    "subcategory::contrast_privacy",
)

# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("xstest", _XSTEST_SUBSET_METRICS)
