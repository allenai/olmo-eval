"""
EQ-Bench Evaluation Tasks

EQ-Bench v2.1 measures emotional understanding. The model reads a short dialogue and rates,
from 0 to 10, how intensely one character is likely to feel each of four emotions, and its
ratings are compared with reference ratings.

Paper: https://arxiv.org/abs/2312.06281
Reference: https://github.com/EQ-bench/EQ-Bench/tree/337df8abdb8f71dfd3bdcc2cc361d40e3cbb245a

The paper describes v1: 60 questions, ratings normalised to sum to 10 and scored by plain
differences, a critique and revised answer, retries until the attempts exceed five, and a
run failing below 50 parsed answers. This task follows the reference code's defaults, whose
score the code reports as v2.

The questions come from ``pbevan11/EQ-Bench``, a copy of the reference's 171 v2 questions
with the prompts already as the reference sends them by default (``-revise`` off): every
prompt equals the reference's ``remove_revision_instructions`` applied to its
``data/eq_bench_v2_questions_171.json``, and every reference rating matches.

Parsing and scoring follow ``lib/scoring.py``: ``*`` and ``#`` are removed, ``Emotion: N``
pairs are read, and an answer must rate exactly the four reference emotions (ignoring case).
An answer must also give at least one rating above zero, the check of the v1 score that
``lib/eq_bench_utils.py`` uses to decide whether an answer parsed. Each emotion's distance
``d`` from the reference adds 0 when ``d`` is 0, ``6.5 / (1 + e^(-1.2(d - 4)))`` when ``d``
is at most 5, and ``d`` otherwise, and an item scores ``10 - 0.7477 * total``, which can be
negative. The EQ-Bench score is 10 times the mean item score over the answers that parse, so
100 is a perfect match, and the reference calibrates the scale so that random answers score
about 0. As in the reference, answers that do not parse are left out of the score,
``parse_failure_rate`` reports how often that happened, and a run in which fewer than 83.33%
of answers parse has no score: every output gets a scoring error and the task fails.

Deviations from the reference:

- When an answer does not parse, the reference asks again up to twice more, raising the
  temperature by 0.15 each time. Each prompt is asked once here.
- The reference's transformers and oobabooga engines also set ``min_p`` 0.1, which is not
  set here.
- Reasoning traces are removed before parsing. An output with a ``<think>`` that never
  closed gives no answer; a trace cut short after a ``<think>`` that the chat template
  wrote is not detected and is parsed as the answer.
- The 83.33% threshold counts the answers received; the reference counts every question, so
  an instance whose generation failed counts there and here counts against the failure
  budget instead. With several samples per prompt, each prompt's mean is averaged; the
  reference averages the scores of whole repeated runs.
- The score is not rounded; the reference rounds it to two decimals.

The ``thinking`` variant samples with temperature 0.6, top_p 0.95 and no token limit, for
reasoning models that cannot finish within the reference's 60 tokens.

Example command to run:

olmo-eval run -m allenai/Olmo-3-7B-Instruct -t eq_bench
"""

import ast
import logging
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers import Scorer, ScoringIncompleteError
from olmo_eval.common.scorers.execution import format_scoring_error
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.base import _store_output_score
from olmo_eval.evals.tasks.common.judge import JudgeScorer
from olmo_eval.evals.tasks.omniscience import extract_final_answer

logger = logging.getLogger(__name__)

EQ_BENCH_REPO = "pbevan11/EQ-Bench"
EQ_BENCH_REVISION = "9ce8e5ffc1a36be5f946b37610ec8c516871f0d3"

# The reference's settings without the revision step (eq-bench.py, lib/eq_bench_utils.py)
DEFAULT_SAMPLING = SamplingParams(max_tokens=60, temperature=0.01)
# Room for reasoning models to finish thinking before they answer
THINKING_SAMPLING = SamplingParams(max_tokens=None, temperature=0.6, top_p=0.95)

# The reference fails a run in which fewer answers parse (lib/run_bench.py)
MIN_PARSEABLE_FRACTION = 0.8333
# Chosen by the reference so that random answers score about zero
ADJUST_CONST = 0.7477

RESULT_KEY = "eq_bench_result"

# =============================================================================
# Parsing and Scoring
# =============================================================================


def parse_ratings(text: str) -> dict[str, str]:
    """Read ``Emotion: N`` ratings as the reference does, after removing ``*`` and ``#``.

    A later rating of the same word replaces an earlier one, and only the digits before
    any decimal point are read.
    """
    text = text.replace("*", "").replace("#", "")
    return dict(re.findall(r"(\w+):\s+(\d+)", text))


def item_score(reference: dict[str, Any], ratings: dict[str, str]) -> float | None:
    """Score ratings against a reference, or return None if the answer does not count.

    An answer counts when it rates exactly the four reference emotions, ignoring case, and
    gives at least one a rating above zero, as the reference's v1 ``calculate_score`` gate
    requires. The score follows ``calculate_score_fullscale`` in ``lib/scoring.py``.
    """
    if len(ratings) != 4:
        return None
    emotions = {reference[f"emotion{i}"].lower(): i for i in range(1, 5)}
    if {emotion.lower() for emotion in ratings} != emotions.keys():
        return None
    if sum(float(rating) for rating in ratings.values()) <= 0:
        return None

    difference_tally = 0.0
    for emotion, rating in ratings.items():
        i = emotions[emotion.lower()]
        d = abs(float(rating) - float(reference[f"emotion{i}_score"]))
        if d == 0:
            scaled = 0.0
        elif d <= 5:
            scaled = 6.5 * (1 / (1 + math.e ** (-1.2 * (d - 4))))
        else:
            scaled = d
        difference_tally += scaled
    return 10 - difference_tally * ADJUST_CONST


def _item_result(output: LMOutput) -> dict[str, Any] | None:
    result = (output.metadata or {}).get(RESULT_KEY)
    return result if isinstance(result, dict) else None


# =============================================================================
# Metrics
# =============================================================================


@dataclass(frozen=True)
class EQBenchScorer(JudgeScorer):
    """Names the score channel of EQ-Bench items, which the task scores itself."""

    name: str = "eq_bench"


@dataclass(frozen=True, slots=True)
class EQBenchScoreMetric(Metric):
    """10 times the mean item score over the answers that parse: 100 matches the reference."""

    name: str = "eq_bench_score"
    scorer: type[Scorer] | Scorer = EQBenchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        scores = [
            result["score"]
            for output in response.outputs
            if (result := _item_result(output)) is not None and result["score"] is not None
        ]
        return 10 * sum(scores) / len(scores) if scores else None

    def supports_scorer_fallback(self) -> bool:
        return False

    def display_format(self) -> str:
        return "raw"


@dataclass(frozen=True, slots=True)
class EQBenchParseFailureRateMetric(Metric):
    """Share of answers that do not parse, which the score leaves out."""

    name: str = "parse_failure_rate"
    scorer: type[Scorer] | Scorer = EQBenchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        results = [r for output in response.outputs if (r := _item_result(output)) is not None]
        if not results:
            return None
        return sum(result["score"] is None for result in results) / len(results)

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"


# =============================================================================
# Task Definition
# =============================================================================


@register("eq_bench")
class EQBench(Task):
    """Rate the emotions of a character in a dialogue, scored against reference ratings."""

    data_source = DataSource(path=EQ_BENCH_REPO, revision=EQ_BENCH_REVISION, split="validation")
    split = Split.VALIDATION
    formatter = ChatFormatter()
    sampling_params = DEFAULT_SAMPLING
    answer_extractor = extract_final_answer
    metrics = (EQBenchScoreMetric(), EQBenchParseFailureRateMetric())
    primary_metric = EQBenchScoreMetric()

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance; references are Python literals."""
        reference = ast.literal_eval(doc["reference_answer_fullscale"])
        return Instance(
            question=doc["prompt"],
            metadata={"id": index + 1, "reference": reference},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Send the prompt as the only user turn, with no system prompt."""
        formatter = self.config.formatter or ChatFormatter()
        return formatter.format(instance, self.get_fewshot())

    async def score_responses(
        self, responses: Sequence[Response], context: Any = None
    ) -> Sequence[Response]:
        """Parse and score every answer, leaving answers that do not parse unscored.

        The runner scores responses one at a time, so the run's parse rate is checked in
        ``compute_metrics``.
        """
        self._extract_answers(responses)
        name = EQBenchScorer.name
        for response in responses:
            reference = response.instance.metadata["reference"]
            item_scores: dict[int, float] = {}
            for index, output in enumerate(response.outputs):
                ratings = parse_ratings(output.extracted_answer or "")
                score = item_score(reference, ratings)
                output.metadata[RESULT_KEY] = {"ratings": ratings, "score": score}
                if score is not None:
                    item_scores[index] = score
                    _store_output_score(output, scorer_name=name, score=score)
            if item_scores:
                response.scores[name] = self._aggregate_output_scores(item_scores)
        return responses

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, dict[str, float]]:
        """Compute the metrics, or none for a run in which too few answers parse.

        The reference reports no score when fewer than 83.33% of answers parse, or when none
        do. Such a run records a scoring error on every output, so the task fails.
        """
        results = [
            result
            for response in responses
            for output in response.outputs
            if (result := _item_result(output)) is not None
        ]
        parsed = sum(result["score"] is not None for result in results)
        if results and parsed / len(results) >= MIN_PARSEABLE_FRACTION:
            return super().compute_metrics(responses)

        message = (
            f"only {parsed} of {len(results)} EQ-Bench answers parse; the reference reports no "
            f"score below {MIN_PARSEABLE_FRACTION:.2%}"
        )
        logger.error(message)
        error = format_scoring_error(ScoringIncompleteError(message), phase="compute_metrics")
        error.pop("infrastructure", None)
        for response in responses:
            for output in response.outputs:
                _store_output_score(
                    output, scorer_name=EQBenchScorer.name, score=None, scoring_error=dict(error)
                )
        return {}


register_variant("eq_bench", "thinking", sampling_params=THINKING_SAMPLING)
