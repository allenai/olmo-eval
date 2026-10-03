"""Held-out multiple-choice items from a training source, asked exactly as in training.

A positive control for supervised finetuning: each JSONL record carries the source's original
user prompt (with its own answer-format instruction) and the teacher's answer letter as gold.
A model finetuned on the source should agree with the teacher more often than its starting
checkpoint does on items it never saw.

Record fields: ``id``, ``prompt``, ``gold`` (letter), ``letters`` (options offered), ``topic``.
The file path is the task's data source; the default is the mount used by the sftlab eval
specs. Answers are read from the text after the reasoning trace with :func:`extract_letter`,
the same function that read the teacher's gold letter.

Registered: ``nemotron_rqa_heldout:cot``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import extract_think_answer
from olmo_eval.evals.tasks.common import Task, register

_LETTER_PATTERNS = (
    r"\(\(\s*\(?([A-J])\)?\s*\)\)",
    r"\\boxed\{\s*\(?([A-J])\)?\s*\}",
    r"answer\s*(?:is|:)\s*\**\s*\(?([A-J])\)?(?![A-Za-z])",
    r"\*\*\s*\(?([A-J])\)?\s*\*\*",
)


def extract_letter(text: str) -> str:
    """The last answer letter stated in one of the source's answer formats, or ''."""
    tail = (text or "")[-600:]
    best, best_pos = "", -1
    for pattern in _LETTER_PATTERNS:
        for match in re.finditer(pattern, tail, flags=re.IGNORECASE):
            if match.start() > best_pos:
                best, best_pos = match.group(1).upper(), match.start()
    return best


@dataclass(frozen=True, slots=True)
class HeldoutLetterScorer(Scorer):
    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        answer = extract_letter(extract_think_answer(output.text or "") or "")
        return 1.0 if answer and answer == str(instance.gold_answer or "").upper() else 0.0


_ACCURACY = AccuracyMetric(name="exact_match", scorer=HeldoutLetterScorer)


@register("nemotron_rqa_heldout:cot")
class NemotronRQAHeldout(Task):
    data_source = DataSource(path="/heldout/nemotron-rqa-heldout.jsonl")
    metrics = (_ACCURACY,)
    primary_metric = _ACCURACY
    sampling_params = SamplingParams(max_tokens=None, temperature=0.6, top_p=0.95)
    num_fewshot = 0
    strip_thinking = True

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if not doc.get("prompt") or not doc.get("gold"):
            return None
        return Instance(
            question=str(doc["prompt"]),
            gold_answer=str(doc["gold"]).upper(),
            choices=tuple(doc.get("letters") or ()),
            metadata={"id": doc.get("id", index), "index": index, "topic": doc.get("topic")},
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_letter(extract_think_answer(output.text or "") or "") or None

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )
