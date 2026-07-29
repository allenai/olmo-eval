"""FrontierScience expert-level scientific reasoning (arXiv 2601.21165).

Two tracks from the open-sourced gold set of ``openai/frontierscience``:

``frontierscience_olympiad``
    100 short-answer olympiad problems (IPhO / IChO / IBO level). A model judge
    compares the attempted answer against the reference answer and returns a
    CORRECT / INCORRECT verdict, so the primary metric is accuracy.

``frontierscience_research``
    60 open-ended PhD-level research sub-tasks. Each carries a rubric totaling
    10 points that credits intermediate derivations as well as the final result.
    A model judge tallies the points earned; the paper marks a response a
    success at seven or more points, which is the primary metric here. The mean
    fraction of rubric points earned is reported alongside it because it is the
    more sensitive signal on a benchmark where frontier models score around 25%.

Both tracks report the primary metric overall and per subject (biology,
chemistry, physics).

The problem statements already end with the benchmark's own answer-format
instruction, so the generation prompt is the ``problem`` field verbatim with
nothing prepended. Olympiad responses are cut down to the text after the final
``FINAL ANSWER`` marker the dataset asks for; research responses are graded
whole, since the rubric credits the derivation.

Judge prompts are reconstructed from the paper's Appendix B listings, including
its ``attemped`` typo. The paper typesets one example as math, so the exact
characters behind ``6.69 ~ 6.7`` are unrecoverable; the rendered form is used.

The paper judges with GPT-5 at high reasoning effort. The deliberate project
default is ``gpt-5.5:high``; set ``OLMO_EVAL_JUDGE=gpt-5:high`` to reproduce the
published numbers, or a cheaper spec such as ``gpt-5-mini`` for iteration. Judge
strictness moves these scores, so cross-model comparisons need one judge spec.

The paper averages Olympiad over 20 independent trials and Research over 30. The
``:paper`` variants request that many samples per problem, and per-instance
scores are the mean over samples rather than the harness default of the max.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import OutputScoreAggregation, Task, register, register_variant
from olmo_eval.evals.tasks.common.base import _store_output_score

logger = logging.getLogger(__name__)

FRONTIERSCIENCE_REPO = "openai/frontierscience"
# Pinned so a Hub update cannot silently change the gold set mid-comparison.
FRONTIERSCIENCE_REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"

FRONTIERSCIENCE_DEFAULT_JUDGE_SPEC = "gpt-5.5:high"
FRONTIERSCIENCE_JUDGE_ATTEMPTS = 3
FRONTIERSCIENCE_DEFAULT_SCORING_CONCURRENCY = 8

FRONTIERSCIENCE_SUBJECTS = ("biology", "chemistry", "physics")

#: Every research rubric in the gold set sums to exactly this many points.
FRONTIERSCIENCE_RUBRIC_TOTAL = 10.0
#: The paper counts a research response as a success at this many rubric points.
FRONTIERSCIENCE_SUCCESS_THRESHOLD = 7.0

#: Scorer channel name shared by every FrontierScience metric.
FRONTIERSCIENCE_SCORER_NAME = "frontierscience_judge"

FRONTIERSCIENCE_OLYMPIAD_JUDGE_PROMPT = (
    "You are grading an attempted answer to a science olympiad problem. You will be given the "
    "problem, attempted answer, and reference answer. Evaluate the solution against the provided "
    "reference solution, ensuring it is complete and matches the reference solution. Pay close "
    "attention to detail and grade it strictly, but fairly.\n\n"
    "The reference answer is either a single number or expression in latex formatting, a chemical "
    "formula, a compound name, or a phrase referring to a specific name, entity, or method.\n\n"
    "Mark the attempted answer as correct if it fully matches the reference answer or is otherwise "
    "equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal "
    "place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a "
    "compound/formula, equivalent when accounting for units, etc.).\n"
    "Mark it as incorrect if it is not equivalent to the reference answer.\n\n"
    "***\n\n"
    "The problem: {problem}\n\n"
    "***\n\n"
    "The reference answer: {reference_answer}\n\n"
    "***\n\n"
    "The attempted answer: {answer}\n\n"
    "***\n\n"
    "First, think step-by-step about whether the attempted answer matches the reference answer.\n"
    'If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your '
    'response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT".'
)

FRONTIERSCIENCE_RESEARCH_JUDGE_PROMPT = (
    "You are grading a science exam.\n\n"
    "You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric "
    "will total up to 10 points.\n\n"
    # "attemped" reproduces the paper's spelling; the prompt is kept verbatim.
    "Evaluate the attemped answer against the provided rubric. Pay close attention to detail and "
    "grade it strictly, but fairly.\n"
    "Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even "
    "if you think the answer is correct but rubric is wrong, you should treat the rubric as the "
    "gold standard).\n"
    "Return the absolute total number of points earned (it can be a decimal based on the "
    "rubric).\n"
    "***\n\n"
    "The problem: {problem}\n\n"
    "***\n\n"
    "The rubric: {rubric}\n\n"
    "***\n\n"
    "The attempted answer: {answer}\n\n"
    "***\n\n"
    "First, think step-by-step about each rubric item. Explain your reasoning for each rubric "
    "item.\n"
    "Then, tally the points up and write `VERDICT: <total_points>` in the last line of your "
    "response, no other text. For example, `VERDICT: 2.5` or `VERDICT: 8`."
)

_OLYMPIAD_VERDICT = re.compile(r"VERDICT:\s*\**\s*(INCORRECT|CORRECT)\b", re.IGNORECASE)
_RESEARCH_VERDICT = re.compile(r"VERDICT:\s*\**\s*`?\s*(-?\d+(?:\.\d+)?)")

# Tolerates the markdown emphasis and trailing colon models add around the marker.
_FINAL_ANSWER_MARKER = re.compile(r"\**\s*FINAL\s+ANSWER\s*\**\s*:?\s*", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop a closed reasoning block so only the visible answer reaches the judge."""
    close_tag = "</think>"
    index = text.rfind(close_tag)
    if index >= 0:
        return text[index + len(close_tag) :].strip()
    return text.strip()


def extract_final_answer(text: str) -> str:
    """Return the text after the benchmark's ``FINAL ANSWER`` marker.

    Falls back to the whole visible response when the marker is absent or
    nothing follows it, so a judge still sees the model's attempt.
    """
    visible = strip_reasoning(text)
    matches = list(_FINAL_ANSWER_MARKER.finditer(visible))
    if not matches:
        return visible
    return visible[matches[-1].end() :].strip() or visible


def parse_olympiad_verdict(raw: str) -> float | None:
    """Map an olympiad judge reply onto 1.0 / 0.0, or None if no verdict parses."""
    matches = list(_OLYMPIAD_VERDICT.finditer(raw))
    if not matches:
        return None
    return 0.0 if matches[-1].group(1).upper() == "INCORRECT" else 1.0


def parse_research_points(raw: str) -> float | None:
    """Read the rubric point total from a research judge reply, clamped to 0-10."""
    matches = list(_RESEARCH_VERDICT.finditer(raw))
    if not matches:
        return None
    points = float(matches[-1].group(1))
    return min(max(points, 0.0), FRONTIERSCIENCE_RUBRIC_TOTAL)


def build_frontierscience_judge_fn(*, scorer_name: str, max_tokens: int) -> JudgeFn:
    """Build the judge from ``$OLMO_EVAL_JUDGE`` or the task-local default."""
    spec = os.getenv("OLMO_EVAL_JUDGE", FRONTIERSCIENCE_DEFAULT_JUDGE_SPEC)
    model, separator, effort = spec.partition(":")
    return build_openai_judge_fn(
        model=model,
        temperature=0.0,
        # A reasoning judge that spends this budget truncates before its verdict
        # line; that call is retried and then scored 0.0.
        max_tokens=max_tokens,
        scorer_name=scorer_name,
        reasoning_effort=(effort if separator else None),
    )


async def judge_with_retries(
    judge_fn: JudgeFn,
    prompt: str,
    parse: Callable[[str], float | None],
) -> tuple[float | None, str]:
    """Call the judge until its verdict parses, returning the value and last reply."""
    raw = ""
    for attempt in range(FRONTIERSCIENCE_JUDGE_ATTEMPTS):
        raw = await judge_fn(prompt)
        value = parse(raw)
        if value is not None:
            return value, raw
        if attempt < FRONTIERSCIENCE_JUDGE_ATTEMPTS - 1:
            await asyncio.sleep(2**attempt)
    return None, raw


@dataclass(frozen=True)
class FrontierScienceScorer(Scorer):
    """Placeholder scorer; FrontierScience scores are computed in ``score_responses``."""

    name: str = FRONTIERSCIENCE_SCORER_NAME

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(f"score:{self.name}", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class FrontierScienceMetric(Metric):
    """Mean of a task-computed response score, optionally restricted to one subject.

    All FrontierScience metrics are proportions on a 0-1 scale, so they display as
    percentages even where the metric name does not follow the framework's naming
    heuristics. A subject slice with no instances in the scored set has nothing to
    average and reports 0.0.
    """

    name: str = ""
    scorer: type[Scorer] | Scorer = FrontierScienceScorer
    score_key: str = ""
    subject: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        selected = [
            response
            for response in responses
            if self.subject is None or response.instance.metadata.get("subject") == self.subject
        ]
        if not selected:
            return 0.0
        key = self.score_key or self.name
        return sum(response.scores.get(key, 0.0) for response in selected) / len(selected)

    def supports_pairwise_scorer_fallback(self) -> bool:
        # The shared scorer channel carries the primary metric only, so a subject
        # slice must never fall back to it.
        return False

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


def _subject_metrics(score_key: str) -> tuple[FrontierScienceMetric, ...]:
    """Build one per-subject metric per FrontierScience subject for a score key."""
    return tuple(
        FrontierScienceMetric(
            name=f"{score_key}_{subject}",
            score_key=score_key,
            subject=subject,
        )
        for subject in FRONTIERSCIENCE_SUBJECTS
    )


OLYMPIAD_ACCURACY = FrontierScienceMetric(name="accuracy", score_key="accuracy")
OLYMPIAD_METRICS = (OLYMPIAD_ACCURACY, *_subject_metrics("accuracy"))

RESEARCH_SUCCESS_RATE = FrontierScienceMetric(name="success_rate", score_key="success_rate")
RESEARCH_RUBRIC_SCORE = FrontierScienceMetric(name="rubric_score", score_key="rubric_score")
RESEARCH_METRICS = (
    RESEARCH_SUCCESS_RATE,
    RESEARCH_RUBRIC_SCORE,
    *_subject_metrics("success_rate"),
    *_subject_metrics("rubric_score"),
)

_OLYMPIAD_SAMPLING = SamplingParams(temperature=0.0, max_tokens=32768)
_RESEARCH_SAMPLING = SamplingParams(temperature=0.0, max_tokens=32768)


class _FrontierScience(Task, ABC):
    """Shared FrontierScience loading, prompting, and judge orchestration."""

    split = Split.TRAIN  # HF JSON files load as a single "train" split
    # The paper averages each problem over independent trials, so multi-sample
    # runs must average rather than take the best sample.
    output_score_aggregation = OutputScoreAggregation.MEAN
    required_secrets = ("OPENAI_API_KEY",)

    #: Response score keys this track populates, primary first.
    score_keys: tuple[str, ...] = ()
    judge_max_tokens: int = 8192

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        problem = str(doc.get("problem") or "").strip()
        answer = str(doc.get("answer") or "").strip()
        if not problem or not answer:
            return None

        subject = str(doc.get("subject") or "").strip().lower()
        if subject and subject not in FRONTIERSCIENCE_SUBJECTS:
            logger.warning("FrontierScience: unexpected subject %r at index %d", subject, index)

        task_group_id = str(doc.get("task_group_id") or "")
        return Instance(
            question=problem,
            gold_answer=answer,
            metadata={
                "id": task_group_id or f"{self.config.name}_{index}",
                "task_group_id": task_group_id,
                "subject": subject,
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Send the problem verbatim; it already carries the answer-format instruction."""
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def _build_judge_fn(self) -> JudgeFn:
        return build_frontierscience_judge_fn(
            scorer_name=type(self).__name__,
            max_tokens=self.judge_max_tokens,
        )

    @abstractmethod
    async def _score_output(
        self,
        response: Response,
        output: LMOutput,
        judge_fn: JudgeFn,
    ) -> dict[str, float]:
        """Judge one output, record its judge details, and return its score keys."""
        ...

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge every output concurrently, then average each instance over its samples."""
        self._extract_answers(responses)
        judge_fn = self._build_judge_fn()

        concurrency = (
            context.scoring_concurrency
            if context is not None
            else FRONTIERSCIENCE_DEFAULT_SCORING_CONCURRENCY
        )
        semaphore = asyncio.Semaphore(max(1, concurrency))

        jobs = [
            (response_index, output_index)
            for response_index, response in enumerate(responses)
            for output_index in range(len(response.outputs))
        ]

        # An instance with no output scores 0.0, which is indistinguishable from a
        # genuine miss unless it is announced: a provider that fails every request
        # otherwise yields a clean all-zero result with no error anywhere.
        missing_output = sum(1 for response in responses if not response.outputs)
        if missing_output:
            logger.warning(
                "FrontierScience has no model output for %d/%d instance(s); each scores 0.0 "
                "without a judge call. If this covers the whole task, generation failed and "
                "the reported score is not a capability measurement.",
                missing_output,
                len(responses),
            )

        async def run(response_index: int, output_index: int) -> dict[str, float]:
            response = responses[response_index]
            async with semaphore:
                return await self._score_output(response, response.outputs[output_index], judge_fn)

        results = await asyncio.gather(*(run(*job) for job in jobs))

        collected: list[dict[str, dict[int, float]]] = [
            {key: {} for key in self.score_keys} for _ in responses
        ]
        for (response_index, output_index), scores in zip(jobs, results, strict=True):
            for key, value in scores.items():
                collected[response_index][key][output_index] = value

        for response, per_key in zip(responses, collected, strict=True):
            for key, output_scores in per_key.items():
                response.scores[key] = self._aggregate_output_scores(output_scores)

        unparsed = sum(
            1
            for response in responses
            for output in response.outputs
            if output.metadata.get("frontierscience_judge_parse_error")
        )
        if unparsed:
            logger.warning(
                "FrontierScience judge returned no parseable verdict for %d/%d output(s) after "
                "%d attempts; those outputs scored 0.0 and stay in the denominator.",
                unparsed,
                len(jobs),
                FRONTIERSCIENCE_JUDGE_ATTEMPTS,
            )
        return responses

    def _record(
        self,
        output: LMOutput,
        *,
        scores: dict[str, float],
        raw_judge_response: str,
        parse_error: bool,
    ) -> None:
        """Persist per-output judge details and score channels."""
        output.metadata["frontierscience_raw_judge_response"] = raw_judge_response
        output.metadata["frontierscience_judge_parse_error"] = parse_error
        for key, value in scores.items():
            output.metadata[f"score:{key}"] = value
        _store_output_score(
            output,
            scorer_name=FRONTIERSCIENCE_SCORER_NAME,
            score=scores[self.score_keys[0]],
        )


@register("frontierscience_olympiad")
class FrontierScienceOlympiad(_FrontierScience):
    """FrontierScience Olympiad track scored by a model equivalence checker."""

    data_source = DataSource(
        path=FRONTIERSCIENCE_REPO,
        data_files="olympiad/test.jsonl",
        revision=FRONTIERSCIENCE_REVISION,
        split="train",
    )
    metrics = OLYMPIAD_METRICS
    primary_metric = OLYMPIAD_ACCURACY
    sampling_params = _OLYMPIAD_SAMPLING
    score_keys = ("accuracy",)
    judge_max_tokens = 8192

    def extract_answer(self, output: LMOutput) -> str:
        """Keep only what follows the dataset's ``FINAL ANSWER`` marker."""
        return extract_final_answer(output.text)

    async def _score_output(
        self,
        response: Response,
        output: LMOutput,
        judge_fn: JudgeFn,
    ) -> dict[str, float]:
        answer = output.extracted_answer
        if not isinstance(answer, str):
            answer = extract_final_answer(output.text)

        prompt = FRONTIERSCIENCE_OLYMPIAD_JUDGE_PROMPT.format(
            problem=response.instance.question,
            reference_answer=response.instance.gold_answer or "",
            answer=answer,
        )
        verdict, raw = await judge_with_retries(judge_fn, prompt, parse_olympiad_verdict)
        scores = {"accuracy": 0.0 if verdict is None else verdict}
        self._record(output, scores=scores, raw_judge_response=raw, parse_error=verdict is None)
        return scores


@register("frontierscience_research")
class FrontierScienceResearch(_FrontierScience):
    """FrontierScience Research track scored against 10-point model-judged rubrics."""

    data_source = DataSource(
        path=FRONTIERSCIENCE_REPO,
        data_files="research/test.jsonl",
        revision=FRONTIERSCIENCE_REVISION,
        split="train",
    )
    metrics = RESEARCH_METRICS
    primary_metric = RESEARCH_SUCCESS_RATE
    sampling_params = _RESEARCH_SAMPLING
    score_keys = ("success_rate", "rubric_score")
    # Rubrics run up to 18 items and the judge reasons through each one before
    # tallying, so the reply needs far more room than the olympiad verdict.
    judge_max_tokens = 32768
    success_threshold: float = FRONTIERSCIENCE_SUCCESS_THRESHOLD

    def extract_answer(self, output: LMOutput) -> str:
        """Grade the whole visible response; the rubric credits the derivation."""
        return strip_reasoning(output.text)

    async def _score_output(
        self,
        response: Response,
        output: LMOutput,
        judge_fn: JudgeFn,
    ) -> dict[str, float]:
        answer = output.extracted_answer
        if not isinstance(answer, str):
            answer = strip_reasoning(output.text)

        prompt = FRONTIERSCIENCE_RESEARCH_JUDGE_PROMPT.format(
            problem=response.instance.question,
            rubric=response.instance.gold_answer or "",
            answer=answer,
        )
        points, raw = await judge_with_retries(judge_fn, prompt, parse_research_points)
        earned = 0.0 if points is None else points
        scores = {
            "success_rate": 1.0 if earned >= self.success_threshold else 0.0,
            "rubric_score": earned / FRONTIERSCIENCE_RUBRIC_TOTAL,
        }
        output.metadata["frontierscience_rubric_points"] = earned
        self._record(output, scores=scores, raw_judge_response=raw, parse_error=points is None)
        return scores


# The paper's scoring protocol: 20 Olympiad trials and 30 Research trials per
# problem, averaged. It does not state a temperature; 1.0 is the API default for
# the models it evaluates and is what makes independent trials differ.
register_variant(
    "frontierscience_olympiad",
    "paper",
    sampling_params=replace(_OLYMPIAD_SAMPLING, temperature=1.0, num_samples=20),
)
register_variant(
    "frontierscience_research",
    "paper",
    sampling_params=replace(_RESEARCH_SAMPLING, temperature=1.0, num_samples=30),
)
