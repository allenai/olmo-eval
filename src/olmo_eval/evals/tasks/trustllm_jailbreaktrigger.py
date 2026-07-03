"""
TrustLLM_JailbreakTrigger Safety Evaluation Task

This module implements the TrustLLM_JailbreakTrigger evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2401.05561

Usage:

    olmo-eval run -m llama3.1-8b -t trustllm_jailbreaktrigger:wg_judge
"""

import logging
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("trustllm_jailbreaktrigger")
class TrustLLM_JailbreakTrigger(SafetyBase):
    """trustllm_jailbreaktrigger safety evaluation task."""

    data_source = DataSource(path="allenai/tulu-3-trustllm-jailbreaktrigger-eval", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        return Instance(
            question=doc["prompt"],
            gold_answer="response_refusal=yes",
            metadata={
                "id": doc["id"],
                "index": index,
                "source": doc["source"],
                "jailbreak_type": doc["label"][0],
            },
        )


_JAILBREAKTRIGGER_SUBSET_METRICS = (
    "source::QBB",
    "source::ITC",
    "jailbreak_type::Refusal sentence prohibition",
    "jailbreak_type::leetspeak",
    "jailbreak_type::bad words",
    "jailbreak_type::programming",
    "jailbreak_type::fixed sentence",
    "jailbreak_type::cot",
    "jailbreak_type::json_format",
    "jailbreak_type::no_punctuation",
    "jailbreak_type::no_long_word",
    "jailbreak_type::without_the",
    "jailbreak_type::url_encode",
    "jailbreak_type::cou",
    "jailbreak_type::scenario",
    "jailbreak_type::multitask",
)


# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("trustllm_jailbreaktrigger", _JAILBREAKTRIGGER_SUBSET_METRICS)
