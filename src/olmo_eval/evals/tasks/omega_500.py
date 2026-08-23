"""OMEGA-500: out-of-distribution math reasoning.

OMEGA (https://arxiv.org/abs/2506.18880) evaluates exploratory,
compositional, and transformative generalization in math. OMEGA-500 is a
500-problem subset spanning the benchmark's sub-categories. Answers are
extracted from the response via a regex cascade and compared to the ground
truth, with a strict metric that demotes answers not stated in the
requested format and a flex metric that accepts them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.answer_extraction import ExtractedAnswer, extract_answer_with_format
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

_BOXED_SUFFIX = "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

_TEMPLATE_SUFFIX = (
    '\n\nShow your work and conclude with "Therefore, the final answer is '
    '\\boxed{answer}." where answer is just the final solution that solves the '
    'problem. E.g., if the answer is 6, conclude with "Therefore, the final '
    'answer is \\boxed{6}."'
)

_ANSWER_FORMAT_REGEX = r"Therefore, the final answer is \\boxed\{(.*)\}"
_PREFIX_REGEXES = (
    r"(?i)Therefore,? the final answer is",
    r"(?i)Therefore,? the answer is",
    r"(?i)the final answer is",
    r"(?i)the answer is",
    r"(?i)answer is",
    r"(?i)answer is:",
    r"(?i)answer:",
)
_ANSWER_REGEXES = (r"[\s\S]*?\\boxed\{(.*?)\}[\s\S]*?", r"(.*)\.?")
_FORMAT_CORRECT_CUTOFF = 0.4

# Mathy delimiters stripped when they surround the whole answer text.
_DELIMITERS_TO_STRIP = (
    ("$", "$"),
    ("\\(", "\\)"),
    ("**", "**"),
    ("***", "***"),
    ("\\[", "\\]"),
    ("\\[\n", "\n\\]"),
)


def _extract(continuation: str) -> ExtractedAnswer:
    """Normalize a continuation and extract the final answer."""
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    res = re.sub(r"\.\s*$", "", output).strip()
    for left, right in _DELIMITERS_TO_STRIP:
        res = re.sub(f"{re.escape(left)}(.*){re.escape(right)}", "\\1", res).strip()
    return extract_answer_with_format(
        res,
        answer_format_regex=_ANSWER_FORMAT_REGEX,
        answer_regexes=_ANSWER_REGEXES,
        prefix_regexes=_PREFIX_REGEXES,
    )


@dataclass(frozen=True, slots=True)
class OmegaExactMatchScorer(Scorer):
    """Case-insensitive exact match on the extracted answer.

    The strict form treats an answer as missing when it was not stated in
    the requested format; the flex form accepts it regardless.
    """

    name: str = "exact_match"
    flex: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        answer, format_correct = _extract(output.text or "")
        if not self.flex and format_correct < _FORMAT_CORRECT_CUTOFF:
            answer = ""
        gold = str(instance.gold_answer or "")
        return 1.0 if answer.lower() == gold.lower() else 0.0


_STRICT = OmegaExactMatchScorer()
_FLEX = OmegaExactMatchScorer(name="exact_match_flex", flex=True)


@register("omega_500")
class Omega500(Task):
    data_source = DataSource(path="saumyamalik/omega-500", split="train")
    split = Split.TRAIN
    metrics = (
        AccuracyMetric(name="exact_match", scorer=_STRICT),
        AccuracyMetric(name="exact_match_flex", scorer=_FLEX),
    )
    primary_metric = AccuracyMetric(name="exact_match_flex", scorer=_FLEX)
    # max_tokens=None generates to the model's context limit, matching the
    # reference regime's effective behavior on any context size.
    sampling_params = SamplingParams(
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
    )

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc["messages"][0]["content"]).replace(_BOXED_SUFFIX, "")
        return Instance(
            question=question + _TEMPLATE_SUFFIX,
            gold_answer=str(doc["ground_truth"]),
            metadata={
                "id": doc.get("index", index),
                "dataset": doc.get("dataset"),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output: LMOutput) -> str:
        return _extract(output.text or "").answer
