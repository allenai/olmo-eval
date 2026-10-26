"""
HalluLens Evaluation Tasks

This module implements HalluLens's short-form extrinsic hallucination tasks: PreciseWikiQA,
and the MixedEntities and GeneratedEntities subtasks of NonExistentRefusal.

- PreciseWikiQA asks short questions answerable from Wikipedia. One judge decides whether a
  response abstains and another whether it gives the gold answer; an answer that is not
  correct is a hallucination.
- MixedEntities asks about species and medicines whose names recombine parts of real names,
  and GeneratedEntities about invented businesses, products and historical events. A judge
  decides whether a response believes the entity exists; one that does is a false acceptance.

Paper: https://arxiv.org/abs/2504.17550
Reference: https://github.com/facebookresearch/HalluLens/tree/80307ac6bc9fd396a38b7a0de4196b931611b728

Prompts, sampling, judge prompts and metric definitions follow the reference. Verdicts are
read with the reference's ``jsonify_ans`` and its exact "correct"/"yes" match. Deviations
from the reference:

- HalluLens generates fresh test sets for each evaluation. These tasks use the static sets
  published as swiss-ai/hallulens, whose columns are those the reference code writes plus a
  ``category`` column. Its GeneratedEntities set has 1,300 prompts whose names come from two
  name-shuffling models, Llama-3.1-70B-Instruct and Mistral-Nemo-Instruct-2407, and asks
  about every name with the same one of the reference's ten templates ("Can you share some
  details about the ..."). The reference code, like the paper, takes 1,950 names by default
  from three name-shuffling models, Llama-3.1-405B-Instruct, GPT-4o and
  Mistral-Nemo-Instruct-2407, and picks a template at random for each.
- The reference judges with Llama-3.1-70B-Instruct (Llama-3.1-8B-Instruct for MixedEntities,
  where the paper reports 70B) served with vLLM. These tasks judge with ``JUDGE_MODEL``
  unless another judge, or a judge panel, is configured.
- Every judge reply with no verdict is asked for again, up to ``JUDGE_ATTEMPTS`` calls, as
  ``jsonify_ans`` does, and an output still without one is left unscored. The reference's
  PreciseWikiQA metrics instead read the first abstention reply with ``json.loads`` and stop
  on one that is not JSON, and its GeneratedEntities retries name a judge it cannot load.
  A verdict must be a JSON boolean, so a reply whose first line is other JSON is asked for
  again; ``jsonify_ans`` keeps it, and the reference's NonExistentRefusal metrics then read
  the truthiness of any value under the key and stop on one without the key. A correctness
  reply with no text, such as a reasoning judge's unclosed trace, is asked for again here;
  the reference counts it as a hallucination.
- Reasoning traces are removed before judging, and the judge sees the answer stripped of
  surrounding whitespace and ``<answer>`` tags. A response with no final answer, such as
  one whose ``<think>`` never closed, counts as an abstention (or a refusal) without a judge
  call; ``no_answer_rate`` reports how often that happened. A trace cut short after a
  ``<think>`` that the chat template wrote is not detected and is judged as the answer.
- The reference forces a "does not believe" verdict for Gemma models on medicines; that
  model-specific rule is not ported.
- Each prompt is answered once; the paper averages three runs, and its NonExistentRefusal
  table also averages MixedEntities and GeneratedEntities, which these tasks report apart.
- The LongWiki task is not implemented.

``correct_rate`` follows the reference code, which judges every response for correctness,
so an abstention the judge accepts counts as correct; the paper counts every refusal as
incorrect.

The ``thinking`` variant samples with temperature 0.6, top_p 0.95 and no token limit, for
reasoning models that cannot finish within the reference's 256 tokens.

Judges: ``-o judge_model=...`` replaces the default judge, and the ``judge_panel`` variant
grades the same outputs with every judge of the shared ``DEFAULT_JUDGE_PANEL`` at once, each
reporting every metric under its own scorer name. ``-o 'judge_models=[...]'`` names any
panel; see ``olmo_eval.evals.tasks.common.judge_panel`` for its entries and settings.

Example commands to run:

olmo-eval beaker launch \
  --harness default \
  -m allenai/Olmo-3-7B-Instruct \
  -t hallulens_precise_wikiqa \
  -w "ai2/olmo-eval-debug" \
  -B "ai2/oe-other" \
  --cluster h100 \
  --inspect

olmo-eval beaker launch \
  --harness default \
  -o auxiliary_providers.qwen_judge.kind=vllm_server \
  -o auxiliary_providers.qwen_judge.model=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  -o auxiliary_providers.kimi_judge.kind=vllm_server \
  -o auxiliary_providers.kimi_judge.model=moonshotai/Kimi-Linear-48B-A3B-Instruct \
  -o auxiliary_providers.kimi_judge.trust_remote_code=true \
  -o auxiliary_providers.kimi_judge.kwargs.tensor_parallel_size=2 \
  -m allenai/Olmo-3-7B-Instruct \
  -t hallulens_precise_wikiqa:judge_panel \
  -w "ai2/olmo-eval-debug" \
  -B "ai2/oe-other" \
  --cluster h100 \
  --gpus 4 \
  --inspect
"""

import asyncio
import json
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.common.judge import JudgeParseError
from olmo_eval.evals.tasks.common.judge_panel import (
    JudgePanelTask,
    JudgeVerdictMetric,
    PanelJudgeScorer,
    register_judge_panel_variant,
    verdict_text,
)
from olmo_eval.evals.tasks.constants.hallulens import (
    ABSTAIN_PROMPT_ANIMAL_NONSENSE,
    ABSTAIN_PROMPT_NONSENSE_MEDICINE,
    ABSTAIN_PROMPT_PLACE_NONSENSE,
    ABSTAIN_PROMPT_UPDATED,
    IS_HALLUCINATION_RESPONSE,
)
from olmo_eval.evals.tasks.omniscience import extract_final_answer

HALLULENS_REPO = "swiss-ai/hallulens"
HALLULENS_REVISION = "815d02b8919b2c24430cbd5f576279e6c0c4fe11"

# The reference samples greedily with a 256-token limit and no system prompt
DEFAULT_SAMPLING = SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0)
# Room for reasoning models to finish thinking before they answer
THINKING_SAMPLING = SamplingParams(max_tokens=None, temperature=0.6, top_p=0.95)

JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
# The reference's judge token limit; it samples the judge greedily, as the OpenAI judge does
JUDGE_MAX_TOKENS = 512
# Judge calls before a reply with no verdict is given up on: jsonify_ans's first reply and
# its four retries (it also records an error for a verdict that comes on the fourth retry)
JUDGE_ATTEMPTS = 5

# JSON keys of the abstention and existence verdicts
ABSTAIN_KEY = "is_abstaining"
BELIEF_KEY = "does_believe"

# Each MixedEntities type's judge prompt, as in the reference's NonsenseMixedEval
MIXED_ENTITY_JUDGE_PROMPTS = {
    "medicine": ABSTAIN_PROMPT_NONSENSE_MEDICINE,
    "animal": ABSTAIN_PROMPT_ANIMAL_NONSENSE,
    "plant": ABSTAIN_PROMPT_ANIMAL_NONSENSE,
    "bacteria": ABSTAIN_PROMPT_ANIMAL_NONSENSE,
}

# =============================================================================
# Judge Verdicts
# =============================================================================


def parse_json_verdict(reply: str, key: str) -> bool | None:
    """Read the true/false verdict a judge gave under ``key``, or None if it gave none.

    As in the reference's ``jsonify_ans``, the reply with spaces removed is searched for
    ``{"<key>":false}`` and then ``{"<key>":true}``, and failing both its first line is read
    as JSON. That JSON must be an object holding a boolean under ``key``, where
    ``jsonify_ans`` accepts any JSON.
    """
    compact = reply.replace(" ", "")
    for verdict in (False, True):
        if json.dumps({key: verdict}, separators=(",", ":")) in compact:
            return verdict
    try:
        parsed = json.loads(reply.split("\n")[0])
    except json.JSONDecodeError:
        return None
    value = parsed.get(key) if isinstance(parsed, dict) else None
    return value if isinstance(value, bool) else None


def is_correct_judgement(reply: str) -> bool:
    """Whether a correctness judge's reply marks the response correct.

    As in the reference, only a reply that is exactly "correct" or "yes", ignoring case,
    does. Any other reply, "unverifiable" included, marks a hallucination.
    """
    return reply.lower() in ("correct", "yes")


def parse_correctness(reply: str) -> bool | None:
    """The correctness verdict of a reply, or None for a reply with no text to judge by."""
    return is_correct_judgement(reply) if reply.strip() else None


# =============================================================================
# Task Scoring
# =============================================================================


@dataclass(frozen=True)
class _HalluLensScorer(PanelJudgeScorer):
    """Panel judge that grades a response's final answer with the reference's prompts."""

    default_model: ClassVar[str] = JUDGE_MODEL
    default_max_tokens: ClassVar[int] = JUDGE_MAX_TOKENS
    attempts: ClassVar[int] = JUDGE_ATTEMPTS


@dataclass(frozen=True)
class PreciseWikiQAScorer(_HalluLensScorer):
    """Judges whether a response abstains and, separately, whether it gives the gold answer.

    Scores 1.0 for a correct answer. As in the reference, both judges see every response,
    so an abstention is judged for correctness too.
    """

    name: str = "precise_wikiqa_judge"

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format the correctness judge's prompt."""
        return self._correctness_prompt(instance, self.final_answer(output) or "")

    def format_abstention_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format the abstention judge's prompt."""
        return self._abstention_prompt(instance, self.final_answer(output) or "")

    @staticmethod
    def _correctness_prompt(instance: Instance, answer: str) -> str:
        return IS_HALLUCINATION_RESPONSE.format(
            prompt=instance.question, generation=answer, gold_answer=instance.gold_answer or ""
        )

    @staticmethod
    def _abstention_prompt(instance: Instance, answer: str) -> str:
        return ABSTAIN_PROMPT_UPDATED.format(prompt=instance.question, generation=answer)

    def parse_judge_response(self, response: str) -> float:
        """Return 1.0 if the correctness judge's reply marks the answer correct, else 0.0.

        Raises:
            ValueError: If the reply has no text after any reasoning.
        """
        correct = parse_correctness(verdict_text(response))
        if correct is None:
            raise ValueError(f"no correctness verdict in judge reply {response!r}")
        return float(correct)

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        """Judge abstention and correctness concurrently, recording both on the instance."""
        answer = self.final_answer(output)
        if answer is None:
            self.record(
                instance,
                {"abstained": True, "correct": False, "empty_response": True},
                raw_response=None,
            )
            return 0.0

        try:
            self._validate_provider(context)
            # Both calls settle before a failure is raised, so neither is left running
            abstention, correctness = await asyncio.gather(
                self._judge(
                    self._abstention_prompt(instance, answer),
                    partial(parse_json_verdict, key=ABSTAIN_KEY),
                    context,
                ),
                self._judge(self._correctness_prompt(instance, answer), parse_correctness, context),
                return_exceptions=True,
            )
            if isinstance(abstention, BaseException):
                raise abstention
            if isinstance(correctness, BaseException):
                raise correctness
        except Exception:
            self.record(instance, None, raw_response=None)
            raise

        abstained, abstention_raw = abstention
        correct, correctness_raw = correctness
        raw_response = {"abstention": abstention_raw, "correctness": correctness_raw}
        if abstained is None or correct is None:
            self.record(instance, None, raw_response=raw_response)
            missing = "abstention" if abstained is None else "correctness"
            raise JudgeParseError(f"no {missing} verdict after {JUDGE_ATTEMPTS} judge attempts")

        self.record(
            instance,
            {"abstained": abstained, "correct": correct, "empty_response": False},
            raw_response=raw_response,
        )
        return float(correct)


@dataclass(frozen=True)
class _NonexistentEntityScorer(_HalluLensScorer):
    """Judges whether a response believes that a nonexistent entity exists.

    Scores 1.0 when it does not, so a refusal scores 1.0.
    """

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format the existence judge's prompt."""
        return self._judge_prompt(instance, self.final_answer(output) or "")

    @abstractmethod
    def _judge_prompt(self, instance: Instance, answer: str) -> str:
        """Format the existence judge's prompt for an already extracted final answer."""
        ...

    def parse_judge_response(self, response: str) -> float:
        """Return 1.0 if the judge says the response does not believe the entity exists.

        Raises:
            ValueError: If the reply holds no verdict.
        """
        believes = parse_json_verdict(verdict_text(response), BELIEF_KEY)
        if believes is None:
            raise ValueError(f"no {BELIEF_KEY!r} verdict in judge reply {response!r}")
        return float(not believes)

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        """Judge whether the response believes the entity exists, recording it on the instance."""
        answer = self.final_answer(output)
        if answer is None:
            self.record(
                instance,
                {"believes": False, "empty_response": True},
                raw_response=None,
            )
            return 1.0

        try:
            self._validate_provider(context)
            believes, raw_response = await self._judge(
                self._judge_prompt(instance, answer),
                partial(parse_json_verdict, key=BELIEF_KEY),
                context,
            )
        except Exception:
            self.record(instance, None, raw_response=None)
            raise

        if believes is None:
            self.record(instance, None, raw_response=raw_response)
            raise JudgeParseError(f"no existence verdict after {JUDGE_ATTEMPTS} judge attempts")

        self.record(
            instance,
            {"believes": believes, "empty_response": False},
            raw_response=raw_response,
        )
        return float(not believes)


@dataclass(frozen=True)
class MixedEntitiesScorer(_NonexistentEntityScorer):
    """Existence judge for species and medicines, with the prompt for the entity's type."""

    name: str = "mixed_entities_judge"

    def _judge_prompt(self, instance: Instance, answer: str) -> str:
        return MIXED_ENTITY_JUDGE_PROMPTS[instance.metadata["type"]].format(
            prompt=instance.question, name=instance.metadata["name"], generation=answer
        )


@dataclass(frozen=True)
class GeneratedEntitiesScorer(_NonexistentEntityScorer):
    """Existence judge for businesses, products and historical events."""

    name: str = "generated_entities_judge"

    def _judge_prompt(self, instance: Instance, answer: str) -> str:
        place = instance.metadata["place"]
        return ABSTAIN_PROMPT_PLACE_NONSENSE.format(
            name=instance.metadata["name"],
            TYPE=instance.metadata["type"],
            PLACE=f" in {place}" if place else "",
            generation=answer,
        )


# =============================================================================
# Metrics
# =============================================================================


@dataclass(frozen=True, slots=True)
class NotRefusedHallucinationRateMetric(JudgeVerdictMetric):
    """Share of the responses that did not abstain whose answer was not correct.

    The paper's hallucination rate when not refused; abstentions are outside the denominator.
    """

    name: str = "any__any__hallucination_rate"
    scorer: type[Scorer] | Scorer = PreciseWikiQAScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return None if judge_result["abstained"] else float(not judge_result["correct"])

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class FalseRefusalRateMetric(JudgeVerdictMetric):
    """Share of responses that abstained.

    Every PreciseWikiQA question is answerable, so every abstention is a false refusal.
    """

    name: str = "any__any__false_refusal_rate"
    scorer: type[Scorer] | Scorer = PreciseWikiQAScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return float(judge_result["abstained"])

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class CorrectRateMetric(JudgeVerdictMetric):
    """Share of all responses whose answer the correctness judge accepted.

    As in the reference code, abstentions are in the denominator and are judged too, so an
    abstention the judge accepts counts as correct; the paper counts every refusal as wrong.
    """

    name: str = "any__any__correct_rate"
    scorer: type[Scorer] | Scorer = PreciseWikiQAScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return float(judge_result["correct"])


@dataclass(frozen=True, slots=True)
class FalseAcceptanceRateMetric(JudgeVerdictMetric):
    """Share of responses that believe a nonexistent entity exists."""

    name: str = "any__any__false_acceptance_rate"
    scorer: type[Scorer] | Scorer = MixedEntitiesScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return float(judge_result["believes"])

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class NoAnswerRateMetric(JudgeVerdictMetric):
    """Share of responses with no final answer, which were graded as abstentions.

    A high value usually means generation stopped at the token limit before reasoning ended.
    """

    name: str = "any__any__no_answer_rate"
    scorer: type[Scorer] | Scorer = PreciseWikiQAScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return float(judge_result["empty_response"])

    def higher_is_better(self) -> bool:
        return False


# =============================================================================
# Task Definitions
# =============================================================================


def _data_source(subset: str) -> DataSource:
    """Return the pinned test split of one swiss-ai/hallulens subset."""
    return DataSource(path=HALLULENS_REPO, subset=subset, revision=HALLULENS_REVISION, split="test")


class _HalluLens(JudgePanelTask):
    """Base for the HalluLens tasks: the prompt as the only chat turn, graded by LLM judges."""

    split = Split.TEST
    sampling_params: SamplingParams = DEFAULT_SAMPLING
    formatter = ChatFormatter()
    answer_extractor = extract_final_answer
    required_secrets = ("OPENAI_API_KEY",)

    judge: ClassVar[PanelJudgeScorer]

    @property
    def request_type(self) -> RequestType:
        """Return the request type for this task."""
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        formatter = self.config.formatter or ChatFormatter()
        return formatter.format(instance, self.get_fewshot())


_PRECISE_WIKIQA_JUDGE = PreciseWikiQAScorer()
_MIXED_ENTITIES_JUDGE = MixedEntitiesScorer()
_GENERATED_ENTITIES_JUDGE = GeneratedEntitiesScorer()


@register("hallulens_precise_wikiqa")
class PreciseWikiQA(_HalluLens):
    """Short fact-seeking questions about Wikipedia pages."""

    data_source = _data_source("precise_wiki")
    judge = _PRECISE_WIKIQA_JUDGE
    metrics = (
        NotRefusedHallucinationRateMetric(scorer=_PRECISE_WIKIQA_JUDGE),
        FalseRefusalRateMetric(scorer=_PRECISE_WIKIQA_JUDGE),
        CorrectRateMetric(scorer=_PRECISE_WIKIQA_JUDGE),
        NoAnswerRateMetric(scorer=_PRECISE_WIKIQA_JUDGE),
    )
    primary_metric = NotRefusedHallucinationRateMetric(scorer=_PRECISE_WIKIQA_JUDGE)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        return Instance(
            question=doc["prompt"],
            gold_answer=doc["answer"],
            metadata={"id": index, "title": doc["title"], "h_score_cat": doc["h_score_cat"]},
        )


@register("hallulens_mixed_entities")
class MixedEntities(_HalluLens):
    """Questions about species and medicines whose names recombine parts of real names."""

    data_source = _data_source("mixed_entities")
    judge = _MIXED_ENTITIES_JUDGE
    metrics = (
        FalseAcceptanceRateMetric(scorer=_MIXED_ENTITIES_JUDGE),
        *(
            FalseAcceptanceRateMetric(
                name=f"type__{entity_type}__false_acceptance_rate", scorer=_MIXED_ENTITIES_JUDGE
            )
            for entity_type in MIXED_ENTITY_JUDGE_PROMPTS
        ),
        NoAnswerRateMetric(scorer=_MIXED_ENTITIES_JUDGE),
    )
    primary_metric = FalseAcceptanceRateMetric(scorer=_MIXED_ENTITIES_JUDGE)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance, rejecting types without a judge prompt."""
        if doc["type"] not in MIXED_ENTITY_JUDGE_PROMPTS:
            raise ValueError(f"MixedEntities row {index} has an unknown type {doc['type']!r}")
        return Instance(
            question=doc["prompt"],
            metadata={"id": index, "name": doc["name"], "type": doc["type"]},
        )


@register("hallulens_generated_entities")
class GeneratedEntities(_HalluLens):
    """Questions about invented businesses, products and historical events."""

    data_source = _data_source("generated_entities")
    judge = _GENERATED_ENTITIES_JUDGE
    metrics = (
        FalseAcceptanceRateMetric(scorer=_GENERATED_ENTITIES_JUDGE),
        NoAnswerRateMetric(scorer=_GENERATED_ENTITIES_JUDGE),
    )
    primary_metric = FalseAcceptanceRateMetric(scorer=_GENERATED_ENTITIES_JUDGE)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance; products have no place."""
        return Instance(
            question=doc["prompt"],
            metadata={
                "id": index,
                "name": doc["name"],
                "type": doc["type_"],
                "place": doc["place"],
                "shuffle_model": doc["shuffle_model"],
            },
        )


# =============================================================================
# Variant Registrations
# =============================================================================

for _task_name in (
    "hallulens_precise_wikiqa",
    "hallulens_mixed_entities",
    "hallulens_generated_entities",
):
    register_variant(_task_name, "thinking", sampling_params=THINKING_SAMPLING)
    register_judge_panel_variant(_task_name)
