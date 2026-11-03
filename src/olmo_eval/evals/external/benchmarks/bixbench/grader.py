"""Open-answer grading, ported from upstream BixBench's ``OpenEndedGrader``.

Follows ``bixbench/graders.py`` of Future-House/BixBench at 49311180 as
``grade_outputs.py`` calls it (``partial_match=True``, ``llm_match=True``):

- ``llm_verifier``: the judge compares the answer with the target.
- ``str_verifier``: answers equal after keeping only lowercased letters and digits
  are correct; otherwise an answer whose cleaned form is a substring of the cleaned
  target is correct; otherwise the judge decides. The substring rule is upstream's,
  so a predicted "1" matches a target of "11".
- ``range_verifier``: the judge decides, with the same prompt as ``llm_verifier``
  and the target range as a "(lower,upper)" string, because upstream passes its
  default template here too.

The answer is graded as submitted, including any ``<answer>`` tags, as upstream
grades ``env.state.answer``. A judge grade other than ``correct`` is incorrect. A
reply with no ``<grade>`` tag is asked again, and one that never has a grade leaves
the answer ungraded, where upstream would count it incorrect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from olmo_eval.common.scorers.execution import ScoringIncompleteError
from olmo_eval.common.scorers.llm_judge import (
    JUDGE_ATTEMPTS,
    ConfiguredJudgeFn,
    Judge,
    JudgeParseError,
    build_openai_judge_fn,
)

from .loader import BixBenchQuestion
from .prompts import OPEN_ENDED_GRADING_PROMPT

logger = logging.getLogger(__name__)

# An alias rather than a dated snapshot: OpenAI publishes no dated gpt-6-luna model,
# so the model behind this name can change. Upstream's grading script defaults to
# gpt-4o at temperature 1.0.
DEFAULT_JUDGE_MODEL = "gpt-6-luna"
DEFAULT_JUDGE_TEMPERATURE = 1.0
DEFAULT_JUDGE_MAX_TOKENS = 8192
JUDGE_RETRIES = 6
JUDGE_RETRY_DELAY = 2.0

_GRADE_PATTERN = re.compile(r"<grade>\s*(.*?)\s*</grade>", re.DOTALL)


@dataclass
class Grade:
    """The grading of one submitted answer.

    ``correct`` is None when the answer could not be graded; ``error`` says why.
    """

    correct: bool | None
    #: ``exact``, ``substring``, ``judge`` or ``no_answer``.
    method: str
    #: The judge's grade word: ``correct``, ``incorrect``, ``refused`` or another word.
    judge_grade: str | None = None
    raw_reply: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_judge_fn(model: str, temperature: float, max_tokens: int) -> ConfiguredJudgeFn:
    return build_openai_judge_fn(
        model=model,
        scorer_name="BixBench judge",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def answer_text(answer: Any) -> str | None:
    """The submitted answer as the string upstream grades, or None if nothing was submitted.

    A dict with one key stands for its value, as upstream's post-processing reads it.
    """
    if answer is None:
        return None
    if isinstance(answer, dict) and len(answer) == 1:
        answer = next(iter(answer.values()))
    if isinstance(answer, str):
        return answer
    return json.dumps(answer) if isinstance(answer, dict | list) else str(answer)


def _clean(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def string_match(target: str, predicted: str) -> str | None:
    """``exact`` or ``substring`` when upstream's string check accepts the answer, else None."""
    cleaned_target, cleaned_predicted = _clean(target), _clean(predicted)
    if cleaned_predicted == cleaned_target:
        return "exact"
    if cleaned_predicted in cleaned_target:
        return "substring"
    return None


def parse_grade(reply: str) -> str | None:
    """The lowercased word between ``<grade>`` tags, or None if the reply has none."""
    match = _GRADE_PATTERN.search(reply)
    return match[1].strip().lower() if match else None


async def grade_answer(
    question: BixBenchQuestion,
    answer: Any,
    judge_fn: ConfiguredJudgeFn,
    limiter: asyncio.Semaphore | None = None,
) -> Grade:
    predicted = answer_text(answer)
    if predicted is None:
        return Grade(correct=False, method="no_answer")
    if question.eval_mode == "str_verifier":
        method = string_match(question.ideal, predicted)
        if method is not None:
            return Grade(correct=True, method=method)

    prompt = OPEN_ENDED_GRADING_PROMPT.format(
        question=question.question, target=question.ideal, predicted=predicted
    )
    judge = Judge(
        call=judge_fn,
        label=f"bixbench {question.question_id}",
        max_retries=JUDGE_RETRIES,
        retry_delay=JUDGE_RETRY_DELAY,
    )
    try:
        grade, raw = await judge.ask(prompt, parse_grade, limiter=limiter)
    except JudgeParseError as e:
        return Grade(
            correct=None,
            method="judge",
            raw_reply=e.raw,
            error=f"judge_parse_error: no grade after {JUDGE_ATTEMPTS} attempts",
        )
    except ScoringIncompleteError as e:
        cause = e.__cause__ or e
        logger.warning(f"Judge call for {question.question_id} failed: {cause}")
        return Grade(correct=None, method="judge", error=f"judge_error: {cause}")
    return Grade(correct=grade == "correct", method="judge", judge_grade=grade, raw_reply=raw)
