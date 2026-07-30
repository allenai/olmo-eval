"""
HarmBench Safety Evaluation Task

This module implements the HarmBench Safety evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2402.04249

Usage:

    olmo-eval run -m llama3.1-8b -t harmbench:wg_judge
"""

import logging
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("harmbench")
class Harmbench(SafetyBase):
    """Harmbench safety evaluation task."""

    data_source = DataSource("allenai/tulu-3-harmbench-eval")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        functional_cat = doc["FunctionalCategory"]
        semantic_cat = doc["SemanticCategory"]

        if functional_cat == "contextual":
            question = doc["ContextString"] + "\n\n---\n\n" + doc["Behavior"]
        else:
            question = doc["Behavior"]

        return Instance(
            question=question,
            gold_answer="response_harmfulness=no",
            metadata={
                "id": doc["BehaviorID"],
                "index": index,
                "functional_category": functional_cat,
                "semantic_category": semantic_cat,
            },
        )


_HARMBENCH_SUBSET_METRICS = (
    "functional_category__standard",
    "functional_category__contextual",
    "functional_category__copyright",
    "semantic_category__copyright",
    "semantic_category__misinformation_disinformation",
    "semantic_category__chemical_biological",
    "semantic_category__illegal",
    "semantic_category__harmful",
    "semantic_category__cybercrime_intrusion",
    "semantic_category__harassment_bullying",
)


# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("harmbench", _HARMBENCH_SUBSET_METRICS)
