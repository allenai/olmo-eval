"""LLM-as-judge grading for HELMET's model-graded tasks.

HELMET grades its LongQA (NarrativeQA) and Summarization tasks with GPT-4o rather
than string overlap, because the references are short phrases or paragraph
summaries that a correct answer can express many ways. This ports the rubrics and
score arithmetic of HELMET's ``scripts/eval_gpt4_longqa.py`` and
``scripts/eval_gpt4_summ.py`` onto :class:`JudgedTask`, so the judge, its retries,
its token budget and the run-wide judge limiter are shared with every other judged
task, and ``-o judge_model=...`` or ``OLMO_EVAL_JUDGE`` changes the judge that runs.

A judge reply that never parses, after retries, leaves the output unscored rather
than scoring it 0.0. Upstream likewise leaves such instances out of the average.

Upstream calls ``gpt-4o-2024-05-13`` through its ``OpenAIModel`` at temperature
0.1, top-p 0.9 and seed 42. OpenAI has retired that snapshot (the API rejects it as
deprecated), so the judge here is ``gpt-4o-2024-08-06``, the one plain ``gpt-4o``
serves, at upstream's temperature. The shared judge client sends no top-p or seed,
and asks again when a reply does not parse where upstream gives up on the first
unparseable reply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar

from olmo_eval.common.scorers.llm_judge import Judge
from olmo_eval.common.types import LMOutput, Response
from olmo_eval.evals.tasks.common.judge import (
    JudgedScoreMetric,
    JudgedTask,
    JudgeScorer,
    TruncationRateMetric,
    visible_response,
)
from olmo_eval.evals.tasks.constants.helmet_summ_prompts import (
    FLUENCY_PROMPT,
    FLUENCY_PROMPT_BOOK,
    PRECISION_PROMPT,
    PRECISION_PROMPT_BOOK,
    RECALL_PROMPT,
    RECALL_PROMPT_BOOK,
)

# Upstream pins gpt-4o-2024-05-13, which the API now rejects as deprecated.
HELMET_JUDGE_MODEL = "gpt-4o-2024-08-06"
HELMET_JUDGE_TEMPERATURE = 0.1
# Upstream's budgets: the OpenAIModel default for LongQA, and the explicit
# generation_max_length eval_gpt4_summ.py sets for summarization.
HELMET_LONGQA_JUDGE_MAX_TOKENS = 2048
HELMET_SUMM_JUDGE_MAX_TOKENS = 4096

# Verbatim from HELMET's scripts/eval_gpt4_longqa.py.
LONGQA_JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.
Although you are not given the context, you will be given a set of correct answers that achieves full scores on all metrics, and you need to assess the provided answers using the correct answers.

Below is your grading rubric:

Fluency:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Correctness:
- Score 0 (Incorrect): The answer does not agree with the provided correct answers at all.
- Score 1 (partly correct): Partly agree with one of the provided correct answers (for example, the question asks for a date and a person; the answer gets the date right but the person wrong).
- Score 2 (correct but not fully relevant): Fully agrees with one of the provided correct answers but mentions other completely irrelevant information. Note that extra details provided in the answer, even if not mentioned in the correct answers, should NOT be seen as irrelevant as long as they are relevant to the question to a reasonable extend.
- Score 3 (correct and relevant): Fully agrees with one of the provided correct answers and only provides information relevant to the question. Note that if the answer is longer than the correct answer, as long as everything in the answer is relevant to the question, it should still be given score 3. For example, if the correct answer is "the North Pole" and the answer is "They are headed for the North Pole", it should still be given a score of 3.

Now, read the following question, answer, and correct answers. First think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"fluency": 0, "correctness": 1}}.

Question: {question}
Correct answers: {correct_answers}
Answer: {parsed_output}
"""  # noqa: E501

# fluency (0-1) x correctness (0-3); dividing by this keeps the score a 0-1
# proportion, and multiplying a reported score back by it recovers HELMET's raw
# "gpt-4-score".
LONGQA_MAX_RAW_SCORE = 3.0

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_JSON_FENCE = re.compile(r"(?:```json)(.+)(?:```)", re.DOTALL)


def parse_judge_json(text: str, *, fenced_fallback: bool = False) -> dict | None:
    """Pull the last JSON object out of a judge reply, as HELMET's ``parse_json`` does.

    The rubrics ask for reasoning followed by a JSON verdict, so the final object is
    the score. ``fenced_fallback`` adds the summarization script's second try, the
    last ```` ```json ```` block, when the last brace-delimited match does not parse.
    """
    matches = _JSON_OBJECT.findall(text)
    if not matches:
        return None
    candidates = [matches[-1]]
    if fenced_fallback:
        candidates += _JSON_FENCE.findall(text)[-1:]
    for candidate in candidates:
        try:
            verdict = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return verdict if isinstance(verdict, dict) else None
    return None


def _numbers(verdict: dict | None, *keys: str) -> tuple[float, ...] | None:
    """Return the verdict's values for ``keys`` as floats, or None if any is unusable."""
    if verdict is None:
        return None
    try:
        return tuple(float(verdict[key]) for key in keys)
    except (KeyError, TypeError, ValueError):
        return None


def parse_longqa_verdict(text: str) -> tuple[float, float] | None:
    """Return a LongQA reply's (fluency, correctness), or None if it gives none."""
    numbers = _numbers(parse_judge_json(text), "fluency", "correctness")
    return None if numbers is None else (numbers[0], numbers[1])


def _summ_parser(*keys: str):
    def parse(text: str) -> tuple[float, ...] | None:
        return _numbers(parse_judge_json(text, fenced_fallback=True), *keys)

    return parse


parse_fluency_verdict = _summ_parser("fluency")
parse_recall_verdict = _summ_parser("recall")
parse_precision_verdict = _summ_parser("precision", "sentence_count")


def parse_longqa_answer(text: str, prefix: str = "Answer:") -> str:
    """The answer HELMET's LongQA judge sees: ``parse_output`` from eval_gpt4_longqa.py.

    Newlines become spaces first, so this is everything after the first ``prefix``,
    or the whole reply when there is none, with an echoed prefix stripped.
    """
    text = text.replace("\n", " ")
    match = re.search(f"(?:{re.escape(prefix)})(.*)", text, flags=re.IGNORECASE)
    answer = (match[1] if match is not None else text).strip()
    return re.sub(f"^{re.escape(prefix)}", "", answer, flags=re.IGNORECASE).strip()


def summ_f1(
    fluency: float, found: float, supported: float, sentence_count: float, num_key_points: int
) -> dict[str, float]:
    """HELMET's fluency-gated F1 over key points, with its recall and precision."""
    recall = found / num_key_points if num_key_points > 0 else 0.0
    precision = supported / sentence_count if sentence_count > 0 else 0.0
    f1 = (
        fluency * 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0.0
    )
    return {
        "gpt4_f1": f1,
        "gpt4_recall": recall,
        "gpt4_precision": precision,
        "gpt4_fluency": fluency,
    }


@dataclass(frozen=True)
class HelmetLongQAJudgeScorer(JudgeScorer):
    """Serves the LongQA judge's stored per-output score."""

    name: str = "gpt4_score"


@dataclass(frozen=True)
class HelmetSummJudgeScorer(JudgeScorer):
    """Serves the summarization judge's stored per-output F1."""

    name: str = "gpt4_f1"


LONGQA_SCORE_METRIC = JudgedScoreMetric(
    name="gpt4_score", scorer=HelmetLongQAJudgeScorer, score_key="gpt4_score"
)
LONGQA_METRICS = (LONGQA_SCORE_METRIC, TruncationRateMetric(scorer=HelmetLongQAJudgeScorer))

SUMM_F1_METRIC = JudgedScoreMetric(
    name="gpt4_f1", scorer=HelmetSummJudgeScorer, score_key="gpt4_f1"
)
SUMM_METRICS = (
    SUMM_F1_METRIC,
    *(
        JudgedScoreMetric(name=key, scorer=HelmetSummJudgeScorer, score_key=key)
        for key in ("gpt4_recall", "gpt4_precision", "gpt4_fluency")
    ),
    TruncationRateMetric(scorer=HelmetSummJudgeScorer),
)


class HelmetLongQAJudgedTask(JudgedTask):
    """HELMET's LongQA judge: fluency x correctness, normalized to [0, 1].

    HELMET's headline ``gpt-4-score`` is the product of a 0/1 fluency judgement and
    a 0-3 correctness judgement, so a disfluent answer scores zero however correct
    it is. The product is divided by its maximum of 3 so the score is a proportion
    like the other HELMET metrics; multiply by 3 to recover HELMET's number.
    """

    judge_model = HELMET_JUDGE_MODEL
    judge_max_tokens = HELMET_LONGQA_JUDGE_MAX_TOKENS
    judge_temperature = HELMET_JUDGE_TEMPERATURE
    judge_scorer_name = HelmetLongQAJudgeScorer.name
    judge_score_key = "gpt4_score"

    def extract_answer(self, output: LMOutput) -> str:
        return parse_longqa_answer(visible_response(output.text))

    async def _score_output(self, response: Response, output: LMOutput, judge: Judge) -> float:
        metadata = response.instance.metadata
        answer = self._judged_answer(output)
        prompt = LONGQA_JUDGE_PROMPT.format(
            question=metadata["judge_question"],
            correct_answers=list(metadata["all_gold_answers"]),
            parsed_output=answer,
        )
        judge_result: dict[str, Any] = {"parsed_output": answer}
        fluency, correctness = await self._judge_output(
            output, judge, prompt, parse_longqa_verdict, judge_result
        )
        judge_result.update(fluency=fluency, correctness=correctness, parse_error=False)
        score = fluency * correctness / LONGQA_MAX_RAW_SCORE
        return self._record_judgement(output, judge_result, {"gpt4_score": score})


class HelmetSummJudgedTask(JudgedTask):
    """HELMET's summarization judge: fluency-gated F1 over key points.

    Three judge calls per summary, run concurrently, because the rubrics grade
    different things and would interfere if merged into one prompt:

    - fluency   -> 0/1, is the summary coherent and non-repetitive
    - recall    -> how many of the pre-extracted key points it covers
    - precision -> how many of its own sentences the expert summary supports

    which combine as HELMET's ``gpt-4-f1``::

        recall = found_key_points / total_key_points
        precision = supported_sentences / sentence_count
        f1 = fluency * 2 * recall * precision / (recall + precision)

    Fluency multiplies rather than averages, so a disfluent summary scores zero
    however much it covers. The key points ship with HELMET's data and reach the
    judge through instance metadata; the loaders refuse data without them.
    ``is_book`` selects the novel-summary rubric variants (infbench_sum_eng) over
    the civil-lawsuit ones (multi_lexsum).
    """

    is_book: ClassVar[bool] = False

    judge_model = HELMET_JUDGE_MODEL
    judge_max_tokens = HELMET_SUMM_JUDGE_MAX_TOKENS
    judge_temperature = HELMET_JUDGE_TEMPERATURE
    judge_scorer_name = HelmetSummJudgeScorer.name
    judge_score_key = "gpt4_f1"
    judge_extra_score_keys = ("gpt4_recall", "gpt4_precision", "gpt4_fluency")

    def _prompts(self, summary: str, key_points: list[str], expert: str) -> tuple[str, str, str]:
        numbered = "\n".join(f"{i + 1}. {kp}" for i, kp in enumerate(key_points))
        fluency, recall, precision = (
            (FLUENCY_PROMPT_BOOK, RECALL_PROMPT_BOOK, PRECISION_PROMPT_BOOK)
            if self.is_book
            else (FLUENCY_PROMPT, RECALL_PROMPT, PRECISION_PROMPT)
        )
        return (
            fluency.format(text=summary),
            recall.format(keypoints=numbered, summary=summary),
            precision.format(expert_summary=expert, summary=summary),
        )

    async def _score_output(self, response: Response, output: LMOutput, judge: Judge) -> float:
        metadata = response.instance.metadata
        key_points = metadata["keypoints"]
        if not key_points:
            raise ValueError(f"{self.config.name} instance {metadata.get('id')} has no key points")
        summary = self._judged_answer(output)
        fluency_prompt, recall_prompt, precision_prompt = self._prompts(
            summary, key_points, metadata["expert_summary"]
        )
        judge_result: dict[str, Any] = {}
        (fluency,), (found,), (supported, sentence_count) = await self._judge_together(
            self._judge_output(
                output,
                judge,
                fluency_prompt,
                parse_fluency_verdict,
                judge_result,
                raw_key="fluency_judge_response",
            ),
            self._judge_output(
                output,
                judge,
                recall_prompt,
                parse_recall_verdict,
                judge_result,
                raw_key="recall_judge_response",
            ),
            self._judge_output(
                output,
                judge,
                precision_prompt,
                parse_precision_verdict,
                judge_result,
                raw_key="precision_judge_response",
            ),
        )
        scores = summ_f1(fluency, found, supported, sentence_count, len(key_points))
        judge_result.update(
            recall_found=found,
            recall_total=len(key_points),
            precision_found=supported,
            precision_total=sentence_count,
            parse_error=False,
        )
        return self._record_judgement(output, judge_result, scores)


class HelmetSummBookJudgedTask(HelmetSummJudgedTask):
    """Summarization judge with HELMET's novel-summary rubrics."""

    is_book = True
