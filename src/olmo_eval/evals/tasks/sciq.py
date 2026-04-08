from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import (
    BPBMetric,
    LogprobMCAccuracyMetric,
    LogprobPerCharMCAccuracyMetric,
)
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.format_helpers import (
    format_mc as _format_mc,
)
from olmo_eval.evals.tasks.common.format_helpers import (
    format_rc as _format_rc,
)


@register("sciq")
class SciQ(Task):
    data_source = DataSource(path="allenai/sciq", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        if self.config.split == Split.ALL:
            if self._instances_cache is None:
                all_instances: list[Instance] = []
                for split in ("test", "validation", "train"):
                    all_instances.extend(self._load_instances(split=split))
                if self.config.limit and len(all_instances) > self.config.limit:
                    all_instances = random.Random(1234).sample(all_instances, self.config.limit)
                self._instances_cache = all_instances
            yield from self._instances_cache
        else:
            yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        choices = [
            doc["distractor1"],
            doc["distractor2"],
            doc["distractor3"],
            doc["correct_answer"],
        ]

        # Only shuffle choices for MC format (matches old oe-eval-internal behavior
        # where RC keeps fixed order: distractors first, correct answer last)
        if self.config.formatter is not None:
            rng = random.Random(index)
            num_choices = len(choices)
            positions = list(range(num_choices))
            rng.shuffle(positions)
            shuffled_choices = [choices[i] for i in positions]
            gold_idx = positions.index(num_choices - 1)
        else:
            shuffled_choices = choices
            gold_idx = 3  # correct_answer is always last

        gold_text = doc["correct_answer"]
        letter = chr(ord("A") + gold_idx)

        return Instance(
            question=question,
            choices=tuple(shuffled_choices),
            gold_answer=gold_text,
            metadata={
                "id": f"sciq_{index}",
                "index": index,
                "dataset": "sciq",
                "gold_idx": gold_idx,
                "gold_text": gold_text,
                "mc_answer": letter,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.num_fewshot == 0:
            return []

        from olmo_eval.data import DataLoader

        loader = DataLoader()
        source = self.config.get_data_source(split=self.fewshot_split)

        all_instances = [
            inst
            for idx, doc in enumerate(loader.load(source))
            if (inst := self.process_doc(doc, idx)) is not None
        ]

        if not all_instances:
            return []

        rng = random.Random(self.config.fewshot_seed)
        return rng.sample(all_instances, min(self.config.num_fewshot, len(all_instances)))

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

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
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


register_variant("sciq", "rc")
register_variant("sciq", "mc", formatter=MultipleChoiceFormatter())
register_variant(
    "sciq",
    "olmo3base",
    num_fewshot=5,
    split=Split.VALIDATION,
    metrics=(LogprobPerCharMCAccuracyMetric(),),
    fewshot_seed=1234,
)


@register("sciq:bpb")
class SciQBPB(SciQ):
    metrics = (BPBMetric(),)

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()

        parts: list[str] = []
        for ex in fewshot:
            answer = ex.gold_answer or ex.metadata.get("gold_text", "")
            parts.append(_format_rc(ex.question, answer))

        parts.append(_format_rc(instance.question))
        prompt = "\n\n".join(parts)

        # Only compute BPB on the gold answer (matches old suite's compute_gold_bpb=True)
        gold_idx = instance.metadata.get("gold_idx", 3)
        choices = instance.choices or ()
        gold_text = choices[gold_idx] if choices else (instance.gold_answer or "")

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=(f" {gold_text}",),
        )


register_variant(
    "sciq:bpb",
    "olmo3base",
    num_fewshot=5,
    split=Split.VALIDATION,
    fewshot_seed=1234,
)
