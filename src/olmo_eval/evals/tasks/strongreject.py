"""
StrongReject Safety Evaluation Task

This module implements the StrongReject evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2402.10260

Usage:

    olmo-eval run -m llama3.1-8b -t strongreject:wg_judge
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


@register("strongreject")
class StrongReject(Task):
    """strongreject safety evaluation task."""

    data_source = DataSource(path="allenai/olmo-eval-strongreject", split="test")
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
            question=doc.get("jailbroken_prompt"),
            gold_answer="response_harmfulness=no",  # used for openai judge only
            metadata={
                "id": doc.get("id"),
                "index": index,
                "jailbreak": doc.get("jailbreak"),
                "category": doc.get("category"),
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
    "category::Disinformation and deception",
    "category::Hate, harassment and discrimination",
    "category::Illegal goods and services",
    "category::Non-violent crimes",
    "category::Sexual content",
    "category::Violence",
    "jailbreak::aim",
    "jailbreak::auto_obfuscation",
    "jailbreak::auto_payload_splitting",
    "jailbreak::base64",
    "jailbreak::base64_input_only",
    "jailbreak::base64_output_only",
    "jailbreak::base64_raw",
    "jailbreak::combination_2",
    "jailbreak::combination_3",
    "jailbreak::dev_mode_v2",
    "jailbreak::dev_mode_with_rant",
    "jailbreak::disemvowel",
    "jailbreak::distractors",
    "jailbreak::distractors_negated",
    "jailbreak::evil_confidant",
    "jailbreak::evil_system_prompt",
    "jailbreak::few_shot_json",
    "jailbreak::gcg_transfer_harmbench",
    "jailbreak::gcg_transfer_universal_attacks",
    "jailbreak::none",
    "jailbreak::pair",
    "jailbreak::pap_authority_endorsement",
    "jailbreak::pap_evidence_based_persuasion",
    "jailbreak::pap_expert_endorsement",
    "jailbreak::pap_logical_appeal",
    "jailbreak::pap_misrepresentation",
    "jailbreak::poems",
    "jailbreak::prefix_injection",
    "jailbreak::refusal_suppression",
    "jailbreak::rot_13",
    "jailbreak::style_injection_json",
    "jailbreak::style_injection_short",
    "jailbreak::translation_guarani",
    "jailbreak::translation_hmong",
    "jailbreak::translation_scotts_gaelic",
    "jailbreak::translation_zulu",
    "jailbreak::wikipedia",
    "jailbreak::wikipedia_with_title",
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
    "strongreject",
    "openai_judge",
    metrics=_safety_metrics(SafetyScorer),
    primary_metric=AccuracyMetric(scorer=SafetyScorer),
    sampling_params=_JUDGE_SAMPLING,
)

# Wildguard judge variant - uses a local auxiliary provider (auxiliary_providers.wg_judge)
# _SR_SCORER = SafetyScorer(
#     provider_name="sr_judge",
#     judge_format="strongreject",
#     judge_request_type=RequestType.LOGLIKELIHOOD,
# )

# _SR_SCORER = SafetyScorer(
#     provider_name="wg_judge",
#     judge_format="wildguard",
#     judge_request_type=RequestType.COMPLETION,
# )

_SR_SCORER = SafetyScorer(
    provider_name="sr_judge",
    judge_format="strongreject",
    judge_request_type=RequestType.LOGLIKELIHOOD,
)

register_variant(
    "strongreject",
    "sr_judge",
    metrics=_safety_metrics(_SR_SCORER),
    primary_metric=AccuracyMetric(scorer=_SR_SCORER),
    sampling_params=_JUDGE_SAMPLING,
)

register_variant(
    "strongreject",
    "sr_judge_thinking",
    metrics=_safety_metrics(_SR_SCORER),
    primary_metric=AccuracyMetric(scorer=_SR_SCORER),
    sampling_params=_JUDGE_SAMPLING,
    answer_extractor=extract_think_answer_only,
)

register_variant(
    "strongreject",
    "base",
    metrics=_safety_metrics(_SR_SCORER),
    primary_metric=AccuracyMetric(scorer=_SR_SCORER),
    sampling_params=_JUDGE_SAMPLING,
    formatter=CompletionFormatter(),
    judge_request_type=RequestType.COMPLETION,
)
