"""MMLU chain-of-thought tasks for the post-training regime.

The base ``mmlu_{subject}`` tasks score by continuation log-probability,
which measures multiple-choice ranking rather than reasoning. These tasks
ask the model to reason in chat format and state a letter, then extract it
with the shared regex cascade — matching the configuration used to report
chain-of-thought MMLU for post-trained models.

One task per subject (``mmlu_abstract_algebra:cot``, ...); the
``mmlu:cot`` suite aggregates all 57.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.answer_extraction import (
    OLMO_3_ANSWER_REGEX_TEMPLATES,
    extract_answer_with_format,
)
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.evals.tasks.mmlu import DEFAULT_MMLU_PATH, MMLU_SUBJECTS

_CHOICE_LABELS = ("A", "B", "C", "D", "E")

_ANSWER_REGEXES = (r"\(?([A-D])\)?",)

_COT_SAMPLING = SamplingParams(
    max_tokens=None,
    temperature=0.6,
    top_p=0.95,
)


def _subject_text(subject: str) -> str:
    return subject.replace("_", " ")


def _description(subject: str) -> str:
    return (
        f"The following are multiple choice questions about {_subject_text(subject)}. "
        "Summarize your reasoning concisely, then conclude with "
        "'Therefore, the answer is: X' where X is one of A, B, C, or D.\n\n"
    )


def _mcq_query(question: str, choices: list[str]) -> str:
    """Question with lettered choices, no trailing answer prefix."""
    choices_text = "\n".join(
        f" {label}. {text}" for label, text in zip(_CHOICE_LABELS, choices, strict=False)
    )
    return f"Question: {question}\n{choices_text}\n"


def _extract_letter(text: str) -> str:
    answer = extract_answer_with_format(
        text,
        answer_regexes=_ANSWER_REGEXES,
        answer_regexes_templates=OLMO_3_ANSWER_REGEX_TEMPLATES,
    ).answer
    return re.sub(r"\(|\)", "", answer)


@dataclass(frozen=True, slots=True)
class MMLUCoTExactMatchScorer(Scorer):
    """Case-insensitive exact match on the extracted answer letter."""

    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        answer = _extract_letter(output.text or "")
        gold = str(instance.gold_answer or "")
        return 1.0 if answer.upper() == gold.upper() else 0.0


_ACCURACY = AccuracyMetric(name="exact_match", scorer=MMLUCoTExactMatchScorer)


class MMLUCoTTask(Task):
    """0-shot chain-of-thought MMLU for one subject."""

    split = Split.TEST
    metrics = (_ACCURACY,)
    primary_metric = _ACCURACY
    sampling_params = _COT_SAMPLING
    num_fewshot = 0

    subject: str

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question", ""))
        choices = list(doc.get("choices") or [])
        if not question or not choices:
            return None
        answer = doc.get("answer")
        if not isinstance(answer, int) or not 0 <= answer < len(_CHOICE_LABELS):
            return None

        return Instance(
            question=_description(self.subject) + _mcq_query(question, choices),
            gold_answer=_CHOICE_LABELS[answer],
            choices=tuple(str(c) for c in choices),
            metadata={
                "id": doc.get("question_id", index),
                "index": index,
                "subject": self.subject,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_letter(output.text or "") or None


for _subject in MMLU_SUBJECTS:
    _name = f"MMLUCoT{_subject.title().replace('_', '')}"
    _cls = type(
        _name,
        (MMLUCoTTask,),
        {
            "__module__": __name__,
            "__qualname__": _name,
            "subject": _subject,
            "data_source": DataSource(path=DEFAULT_MMLU_PATH, subset=_subject),
        },
    )
    globals()[_name] = _cls
    register(f"mmlu_{_subject}:cot")(_cls)
