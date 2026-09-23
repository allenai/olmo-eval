"""
AA-Omniscience Gemini judge comparison

Registers the ``omniscience:judge_gemini_compare`` variant, which grades every
response with both the default OpenAI judge and a Gemini judge so the two can be
compared on identical outputs. Gemini grades are stored under separate metadata
keys and reported under the ``omniscience_judge_gemini`` scorer, alongside a
judge agreement metric.

Example commands to run:
    uv run olmo-eval run -m <model> -t omniscience:judge_gemini_compare
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.metrics import AccuracyMetric, Metric, SubsetAccuracyMetric
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.tasks.common import register_variant
from olmo_eval.evals.tasks.omniscience import (
    GRADE_INDEX_POINTS,
    SUBSET_METRICS,
    HallucinationRateMetric,
    OmniscienceIndexMetric,
    OmniscienceScorer,
    scorer,
)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_RESULT_KEY = "gemini_judge_result"
GEMINI_PARSING_ERROR_KEY = "gemini_is_parsing_error"


def build_gemini_judge_fn(
    model: str = "gemini-2.5-flash",
    scorer_name: str = "GeminiOmniscienceScorer",
    max_tokens: int = 16384,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    max_retries: int = 5,
) -> JudgeFn:
    """Build a lazy async judge function using Gemini's OpenAI-compatible API.

    The returned function validates GEMINI_API_KEY on first call, not at construction.

    Args:
        model: Gemini model to use for judging.
        scorer_name: Name of the scorer class (for error messages).
        max_tokens: Maximum tokens in the judge response, including thinking tokens.
        temperature: Sampling temperature for the judge.
        reasoning_effort: Optional thinking level for the judge ("none" disables
            thinking); when omitted, the model's default thinking behavior is used.
        max_retries: Client-level retries for rate limits and transient errors.

    Returns:
        An async judge function that validates and calls Gemini.
    """
    _client: list = []

    async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
        if not _client:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    f"GEMINI_API_KEY environment variable is required for {scorer_name}."
                )

            from openai import AsyncOpenAI

            _client.append(
                AsyncOpenAI(
                    api_key=api_key,
                    base_url=GEMINI_OPENAI_BASE_URL,
                    max_retries=max_retries,
                )
            )

        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        response = await _client[0].chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    return judge


@dataclass(frozen=True)
class GeminiOmniscienceScorer(OmniscienceScorer):
    """Omniscience grader backed by a Gemini judge.

    Grades are recorded under Gemini-specific metadata keys so they can sit
    alongside the default judge's grades on the same instance.
    """

    name: str = "omniscience_judge_gemini"
    judge_fn: JudgeFn = field(default_factory=build_gemini_judge_fn)

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        """Score with the Gemini judge without touching the default judge's metadata."""
        view = replace(instance, metadata={})
        try:
            return await super().ascore_with_context(view, output, context)
        finally:
            instance.metadata[GEMINI_RESULT_KEY] = view.metadata.get("judge_result")
            instance.metadata[GEMINI_PARSING_ERROR_KEY] = view.metadata.get(
                "is_parsing_error", True
            )


def _with_gemini_grades(response: Response) -> Response:
    """Return a view of the response whose judge grade is the Gemini grade."""
    metadata = {
        **response.instance.metadata,
        "judge_result": response.instance.metadata.get(GEMINI_RESULT_KEY),
    }
    return replace(response, instance=replace(response.instance, metadata=metadata))


@dataclass(frozen=True, slots=True)
class GeminiOmniscienceIndexMetric(OmniscienceIndexMetric):
    """Omniscience index computed from the Gemini judge's grades."""

    def compute(self, responses: Sequence[Response]) -> float:
        return OmniscienceIndexMetric.compute(self, [_with_gemini_grades(r) for r in responses])

    def compute_instance(self, response: Response) -> float | None:
        return OmniscienceIndexMetric.compute_instance(self, _with_gemini_grades(response))


@dataclass(frozen=True, slots=True)
class GeminiHallucinationRateMetric(HallucinationRateMetric):
    """Hallucination rate computed from the Gemini judge's grades."""

    def compute(self, responses: Sequence[Response]) -> float:
        return HallucinationRateMetric.compute(self, [_with_gemini_grades(r) for r in responses])

    def compute_instance(self, response: Response) -> float | None:
        return HallucinationRateMetric.compute_instance(self, _with_gemini_grades(response))


@dataclass(frozen=True, slots=True)
class JudgeAgreementMetric(Metric):
    """Share of instances where the default and Gemini judges assign the same grade."""

    name: str = "judge_agreement"
    scorer: type[Scorer] | Scorer = GeminiOmniscienceScorer

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        default_grade = response.instance.metadata.get("judge_result")
        gemini_grade = response.instance.metadata.get(GEMINI_RESULT_KEY)
        if default_grade not in GRADE_INDEX_POINTS or gemini_grade not in GRADE_INDEX_POINTS:
            return None
        return float(default_grade == gemini_grade)

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


gemini_scorer = GeminiOmniscienceScorer()

register_variant(
    "omniscience",
    "judge_gemini_compare",
    metrics=(
        OmniscienceIndexMetric(scorer=scorer),
        HallucinationRateMetric(scorer=scorer),
        AccuracyMetric(scorer=scorer),
        *(SubsetAccuracyMetric(name=name, scorer=scorer) for name in SUBSET_METRICS),
        GeminiOmniscienceIndexMetric(scorer=gemini_scorer),
        GeminiHallucinationRateMetric(scorer=gemini_scorer),
        AccuracyMetric(scorer=gemini_scorer),
        *(SubsetAccuracyMetric(name=name, scorer=gemini_scorer) for name in SUBSET_METRICS),
        JudgeAgreementMetric(scorer=gemini_scorer),
    ),
    primary_metric=OmniscienceIndexMetric(scorer=scorer),
    required_secrets=("OPENAI_API_KEY", "GEMINI_API_KEY"),
)
