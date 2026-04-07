from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter, PPLFormatter
from olmo_eval.common.metrics import (
    BPBMetric,
    LogprobMCAccuracyMetric,
    LogprobPerCharMCAccuracyMetric,
    LogprobUncondMCAccuracyMetric,
)
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_regime, register_variant
from olmo_eval.evals.tasks.common.format_helpers import format_mc as _format_mc
from olmo_eval.evals.tasks.common.format_helpers import format_rc as _format_rc
from olmo_eval.evals.tasks.constants.arc import (
    ARC_CHALLENGE_FIXED_FEWSHOT,
    ARC_EASY_FIXED_FEWSHOT,
)

_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def _process_arc_doc(doc: dict[str, Any], index: int, dataset: str) -> Instance | None:
    question = doc.get("question", "")
    if not question:
        return None

    choices_data = doc.get("choices", {})
    choices = choices_data.get("text", [])
    if not choices:
        return None

    answer_key = doc.get("answerKey", "")
    letter = _NUM_TO_LETTER.get(answer_key, answer_key)
    gold_idx = ord(letter) - ord("A") if letter else 0
    gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

    return Instance(
        question=question,
        choices=tuple(choices),
        gold_answer=letter,
        metadata={
            "id": doc.get("id", f"{dataset}_{index}"),
            "index": index,
            "dataset": dataset,
            "gold_idx": gold_idx,
            "gold_text": gold_text,
            "num_choices": len(choices),
        },
    )


def _build_arc_fixed_fewshot(
    raw_docs: list[dict[str, Any]], num_fewshot: int, seed: int
) -> list[Instance]:
    instances = []
    for doc in raw_docs:
        question = doc["question"]
        choices = tuple(doc["choices"]["text"])
        answer_key = doc["answerKey"]
        letter = _NUM_TO_LETTER.get(answer_key, answer_key)
        gold_idx = ord(letter) - ord("A")
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        instances.append(
            Instance(
                question=question,
                choices=choices,
                gold_answer=gold_text,
                metadata={
                    "gold_idx": gold_idx,
                    "gold_text": gold_text,
                    "mc_answer": letter,
                },
            )
        )

    if num_fewshot and num_fewshot < len(instances):
        instances = instances[:num_fewshot]
    return instances


class _ARCBase(Task):
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    _fewshot_data: list[dict[str, Any]] = []
    _fewshot_source_name: str = ""
    _dataset_name: str = ""

    @property
    def instances(self) -> Iterator[Instance]:
        if self.config.split == Split.ALL:
            if self._instances_cache is None:
                all_instances: list[Instance] = []
                for split in ("test", "validation", "train"):
                    all_instances.extend(self._load_instances(split=split))
                self._instances_cache = all_instances
            yield from self._instances_cache
        else:
            yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return _process_arc_doc(doc, index, self._dataset_name)

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == self._fewshot_source_name:
            return _build_arc_fixed_fewshot(
                self._fewshot_data, self.config.num_fewshot, self.config.fewshot_seed
            )
        return super()._build_fewshot()

    def _uses_uncond_metric(self) -> bool:
        """Check if any configured metric requires unconditional normalization."""
        return any(isinstance(m, LogprobUncondMCAccuracyMetric) for m in self.config.metrics)

    def _is_bpb(self) -> bool:
        return any(isinstance(m, BPBMetric) for m in self.config.metrics)

    def _format_bpb_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        parts: list[str] = []
        for ex in fewshot:
            answer = ex.gold_answer or ex.metadata.get("gold_text", "")
            parts.append(_format_rc(ex.question, answer))
        parts.append(_format_rc(instance.question))
        prompt = "\n\n".join(parts)

        gold_idx = instance.metadata.get("gold_idx", 0)
        gold_text = (
            instance.choices[gold_idx]
            if instance.choices and 0 <= gold_idx < len(instance.choices)
            else instance.gold_answer
        )
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=(f" {gold_text}",),
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self._is_bpb():
            return self._format_bpb_request(instance)

        fewshot = self.get_fewshot()
        is_mc = isinstance(self.config.formatter, MultipleChoiceFormatter)

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.metadata.get("mc_answer", "")
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
                answer = ex.gold_answer or ex.metadata.get("gold_text", "")
                parts.append(_format_rc(ex.question, answer))

        if is_mc:
            parts.append(_format_mc(instance.question, instance.choices or ()))
            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
        else:
            parts.append(_format_rc(instance.question))
            continuations = tuple(f" {c}" for c in (instance.choices or ()))

        prompt = "\n\n".join(parts)

        # For unconditional normalization (acc_uncond), generate both conditioned
        # and unconditional continuations using continuation_prompts.
        if not is_mc and self._uses_uncond_metric():
            uncond_prompt = "Answer:"
            num_choices = len(continuations)
            # Double continuations: first N conditioned, next N unconditional
            all_continuations = continuations + continuations
            # Per-continuation prompts: conditioned use full context, uncond use "Answer:"
            all_cont_prompts = tuple([prompt] * num_choices + [uncond_prompt] * num_choices)
            return LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt=prompt,
                continuations=all_continuations,
                continuation_prompts=all_cont_prompts,
            )

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


@register("arc_easy")
class ARCEasy(_ARCBase):
    data_source = DataSource(path="allenai/ai2_arc", subset="ARC-Easy", split="test")
    split = Split.TEST
    _fewshot_data = ARC_EASY_FIXED_FEWSHOT
    _fewshot_source_name = "olmes_arc_easy_fixed"
    _dataset_name = "arc_easy"


@register("arc_challenge")
class ARCChallenge(_ARCBase):
    data_source = DataSource(path="allenai/ai2_arc", subset="ARC-Challenge", split="test")
    split = Split.TEST
    _fewshot_data = ARC_CHALLENGE_FIXED_FEWSHOT
    _fewshot_source_name = "olmes_arc_challenge_fixed"
    _dataset_name = "arc_challenge"


register_variant("arc_easy", "rc")
register_variant("arc_easy", "mc", formatter=MultipleChoiceFormatter())
register_variant("arc_easy", "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(),))
register_variant(
    "arc_easy",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_arc_easy_fixed",
    split=Split.ALL,
    metrics=(LogprobPerCharMCAccuracyMetric(),),
)
register_regime(
    "arc_easy",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_arc_easy_fixed",
    split=Split.ALL,
)
register_variant("arc_easy", "olmes", num_fewshot=5, fewshot_source="olmes_arc_easy_fixed")
register_variant("arc_easy", "full")

register_variant("arc_challenge", "rc")
register_variant("arc_challenge", "mc", formatter=MultipleChoiceFormatter())
register_variant("arc_challenge", "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(),))
register_variant(
    "arc_challenge",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_arc_challenge_fixed",
    split=Split.ALL,
    metrics=(LogprobUncondMCAccuracyMetric(),),
)
register_regime(
    "arc_challenge",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_arc_challenge_fixed",
    split=Split.ALL,
)
register_variant(
    "arc_challenge",
    "olmes",
    num_fewshot=5,
    fewshot_source="olmes_arc_challenge_fixed",
)
register_variant("arc_challenge", "full")
