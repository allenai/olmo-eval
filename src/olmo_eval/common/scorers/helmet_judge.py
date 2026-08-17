"""LLM-as-judge scorers for HELMET's model-graded tasks.

HELMET grades its LongQA and Summarization tasks with GPT-4 rather than string
overlap, because the references are short phrases or paragraph summaries that
a correct answer can express many ways. This ports the LongQA rubric from
HELMET's `scripts/eval_gpt4_longqa.py`; the judge model and temperature match
upstream so scores stay comparable to published numbers.
"""

import json
import re
from dataclasses import dataclass, field

from olmo_eval.common.scorers.llm_judge import JudgeFn, LLMJudgeScorer, build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput

# HELMET grades with gpt-4o at temperature 0.1 (scripts/eval_gpt4_longqa.py).
HELMET_JUDGE_MODEL = "gpt-4o-2024-05-13"
HELMET_JUDGE_TEMPERATURE = 0.1
# The rubric asks the judge to reason step by step before emitting JSON, so it
# needs far more room than the 10-token default used for letter-grade judges.
HELMET_JUDGE_MAX_TOKENS = 1024

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

# fluency (0-1) x correctness (0-3); dividing by this keeps the scorer inside
# the [0, 1] contract the base class documents, and multiplying a reported
# score back by it recovers HELMET's raw "gpt-4-score".
LONGQA_MAX_RAW_SCORE = 3.0


def parse_judge_json(text: str) -> dict | None:
    """Pull the last JSON object out of a judge response.

    Mirrors HELMET's `parse_json`: the rubric asks for reasoning followed by a
    JSON verdict, so the final object is the score.
    """
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def _build_helmet_judge_fn(scorer_name: str) -> JudgeFn:
    return build_openai_judge_fn(
        model=HELMET_JUDGE_MODEL,
        scorer_name=scorer_name,
        max_tokens=HELMET_JUDGE_MAX_TOKENS,
        temperature=HELMET_JUDGE_TEMPERATURE,
    )


@dataclass(frozen=True)
class HelmetLongQAJudgeScorer(LLMJudgeScorer):
    """HELMET's LongQA judge: fluency x correctness, normalized to [0, 1].

    HELMET's headline `gpt-4-score` is the product of a 0/1 fluency judgement
    and a 0-3 correctness judgement, so a disfluent answer scores zero however
    correct it is. That product is divided by its maximum of 3 here, both to
    respect the [0, 1] range the base class documents and so this task doesn't
    dominate a suite average against metrics that are already proportions --
    multiply a reported score by 3 to recover HELMET's number.

    An unparseable judge response scores 0.0, matching HELMET, which skips such
    responses when averaging rather than counting them as correct.
    """

    name: str = "gpt4_score"
    judge_fn: JudgeFn = field(
        default_factory=lambda: _build_helmet_judge_fn("HelmetLongQAJudgeScorer")
    )

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        metadata = instance.metadata or {}
        correct_answers = metadata.get("all_gold_answers") or instance.gold_answer or []
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]

        return LONGQA_JUDGE_PROMPT.format(
            question=metadata.get("judge_question", instance.question),
            correct_answers=list(correct_answers),
            parsed_output=output.extracted_answer or output.text,
        )

    def parse_judge_response(self, response: str) -> float:
        scores = parse_judge_json(response)
        if scores is None or "fluency" not in scores or "correctness" not in scores:
            return 0.0
        try:
            fluency = float(scores["fluency"])
            correctness = float(scores["correctness"])
        except (TypeError, ValueError):
            return 0.0
        return (fluency * correctness) / LONGQA_MAX_RAW_SCORE
