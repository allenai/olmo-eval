"""
Toxigen Safety Evaluation Task

This module implements the toxigen evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2203.09509

Usage:

    olmo-eval run -m llama3.1-8b -t toxigen:tox_judge
"""

import logging
from typing import Any

from olmo_eval.common.scorers import SafetyScorer
from olmo_eval.common.types import Instance, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("toxigen")
class Toxigen(SafetyBase):
    """toxigen safety evaluation task."""

    data_source = DataSource(path="allenai/olmo-eval-toxigen", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""

        if not doc["subsample"]:
            return None

        return Instance(
            question=doc["instruction"],
            gold_answer="response_harmfulness=no",  # used for openai varient not original judge
            metadata={
                "id": doc["id"],
                "index": index,
                "target_groups": doc["target_groups"],
            },
        )


toxigen_subsets = (
    "target_groups__asian",
    "target_groups__black",
    "target_groups__chinese",
    "target_groups__jewish",
    "target_groups__latino",
    "target_groups__lgbtq",
    "target_groups__mental_disability",
    "target_groups__mexican",
    "target_groups__middle_east",
    "target_groups__muslim",
    "target_groups__native_american",
    "target_groups__physical_disability",
    "target_groups__trans",
    "target_groups__women",
)


# =============================================================================
# Variant Registrations
# =============================================================================


# tomh/toxigen judge variant - uses a local auxiliary provider (auxiliary_providers.tox_judge)

tox_scorer = SafetyScorer(
    provider_name="wg_judge",
    judge_format="wildguard",
    judge_request_type=RequestType.COMPLETION,
)

register_safety_variants("trustllm_jailbreaktrigger", toxigen_subsets, tox_scorer, "wg_judge")
