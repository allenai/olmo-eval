"""Chain-of-thought chat versions of log-probability multiple-choice tasks.

The base tasks score the log-probability of each option after "Answer:", which a
post-trained reasoning model never gets to reason before. Each ``{task}:cot``
variant here keeps the base task's data loading, filtering and ``limit``
sampling, so it scores the same items, but asks in chat format with lettered
options, lets the model reason, and scores the letter after "Therefore, the
answer is (X)". Prompting, sampling and extraction are the MMLU-Pro
chain-of-thought recipe (``mmlu_pro_{category}:cot``).

Registered: arc_challenge:cot, arc_easy:cot, sciq:cot, medqa_en:cot, piqa:cot,
winogrande:cot, qasper_yesno:cot, sciriff_yesno:cot.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.evals.tasks.arc import ARCChallenge, ARCEasy
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.evals.tasks.medqa_en import MedQAEn
from olmo_eval.evals.tasks.mmlu_pro import (
    _CHOICE_LABELS,
    _COT_ACCURACY,
    _COT_DESCRIPTION,
    _COT_FINAL_DESCRIPTION,
    _COT_SAMPLING,
    _extract_cot_letter,
)
from olmo_eval.evals.tasks.piqa import PiQA
from olmo_eval.evals.tasks.qasper_yesno import QasperYesNo
from olmo_eval.evals.tasks.sciq import SciQ
from olmo_eval.evals.tasks.sciriff_yesno import SciriffYesNo
from olmo_eval.evals.tasks.winogrande import Winogrande


def _options(choices: tuple[str, ...]) -> str:
    return "".join(
        f" ({label}) {text}\n" for label, text in zip(_CHOICE_LABELS, choices, strict=False)
    )


def _plain(instance: Instance) -> str:
    return f"Question: {instance.question}\n"


def _piqa(instance: Instance) -> str:
    return f"Question: Which solution better achieves this goal? {instance.question}\n"


def _winogrande(instance: Instance) -> str:
    sentence = instance.question.replace("_", "___")
    return f"Question: Which option correctly fills the blank? {sentence}\n"


def _with_source(instance: Instance) -> str:
    source = str(instance.metadata.get("source", "")).strip()
    return (f"Passage: {source}\n\n" if source else "") + f"Question: {instance.question}\n"


class _ChatMCMixin:
    """Turns the base task's instance into a lettered chat question."""

    formatter = None
    metrics = (_COT_ACCURACY,)
    primary_metric = _COT_ACCURACY
    sampling_params = _COT_SAMPLING
    num_fewshot = 0
    strip_thinking = True
    answer_extractor = staticmethod(_extract_cot_letter)
    shuffle_choices = False
    stem: Callable[[Instance], str] = staticmethod(_plain)

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        base = super().process_doc(doc, index)  # type: ignore[misc]
        if base is None:
            return None
        choices = tuple(str(c) for c in (base.choices or ()))
        gold = int(base.metadata.get("gold_idx", -1))
        if not choices or not 0 <= gold < len(choices) or len(choices) > len(_CHOICE_LABELS):
            return None
        if self.shuffle_choices:
            # The log-probability base keeps SciQ's correct answer last; shuffle per item, as
            # SciQ's own multiple-choice format does.
            order = list(range(len(choices)))
            random.Random(index).shuffle(order)
            gold = order.index(gold)
            choices = tuple(choices[i] for i in order)
        query = type(self).stem(base) + _options(choices)
        return Instance(
            question=query,
            gold_answer=_CHOICE_LABELS[gold],
            choices=choices,
            metadata={**base.metadata, "gold_idx": gold, "mc_answer": _CHOICE_LABELS[gold]},
        )

    def extract_answer(self, output: Any) -> str | None:
        return _extract_cot_letter(getattr(output, "text", "") or "") or None

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=(
                {
                    "role": "user",
                    "content": _COT_DESCRIPTION + instance.question + _COT_FINAL_DESCRIPTION,
                },
            ),
        )


_VARIANTS: dict[str, tuple[type[Task], Callable[[Instance], str], bool]] = {
    "arc_challenge": (ARCChallenge, _plain, False),
    "arc_easy": (ARCEasy, _plain, False),
    "sciq": (SciQ, _plain, True),
    "medqa_en": (MedQAEn, _plain, False),
    "piqa": (PiQA, _piqa, False),
    "winogrande": (Winogrande, _winogrande, False),
    "qasper_yesno": (QasperYesNo, _with_source, False),
    "sciriff_yesno": (SciriffYesNo, _with_source, False),
}

for _name, (_base, _stem, _shuffle) in _VARIANTS.items():
    _cls_name = _base.__name__ + "ChatCoT"
    _cls = type(
        _cls_name,
        (_ChatMCMixin, _base),
        {
            "__module__": __name__,
            "__qualname__": _cls_name,
            "stem": staticmethod(_stem),
            "shuffle_choices": _shuffle,
        },
    )
    globals()[_cls_name] = _cls
    register(f"{_name}:cot")(_cls)
