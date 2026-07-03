"""
WildGuardTest Safety Evaluation Task

This module implements the WildGuardTest evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2406.18495

Usage:

    olmo-eval run -m llama3.1-8b -t wildguard:wg_judge
"""

import logging
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("wildguardtest")
class WildGuardTest(SafetyBase):
    """wildguardtest safety evaluation task."""

    data_source = DataSource(path="allenai/wildguardmix", subset="wildguardtest", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        # Drop the instance if prompt_harm_label or response_harm_label is NA
        if doc["prompt_harm_label"] is None or doc["response_harm_label"] is None:
            return None

        # Drop examples where the prompt label is not harmful
        if doc["prompt_harm_label"] != "harmful":
            return None

        prompt_type = "adversarial" if doc["adversarial"] else "vanilla"

        question = doc["prompt"]

        return Instance(
            question=question,
            gold_answer="response_harmfulness=no",
            metadata={
                "id": index,
                "subcategory": doc["subcategory"],
                "prompt_type": prompt_type,
            },
        )


_WILDGGUARD_SUBSET_METRICS = (
    "prompt_type::vanilla",
    "prompt_type::adversarial",
    "subcategory::private_information_individual",
    "subcategory::others",
    "subcategory::cyberattack",
    "subcategory::sexual_content",
    "subcategory::social_stereotypes_and_unfair_discrimination",
    "subcategory::toxic_language_hate_speech",
    "subcategory::violence_and_physical_harm",
    "subcategory::defamation_encouraging_unethical_or_unsafe_actions",
    "subcategory::fraud_assisting_illegal_activities",
    "subcategory::mental_health_over-reliance_crisis",
    "subcategory::causing_material_harm_by_disseminating_misinformation",
    "subcategory::disseminating_false_or_misleading_information_encouraging_disinformation_campaigns",
    "subcategory::copyright_violations",
    "subcategory::sensitive_information_organization_government",
)


# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("wildguardtest", _WILDGGUARD_SUBSET_METRICS)
