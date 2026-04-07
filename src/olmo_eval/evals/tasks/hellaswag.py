from __future__ import annotations

import random
import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.hellaswag import HELLASWAG_FIXED_FEWSHOT


def _preprocess(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\.? \[title\]", ". ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _process_hellaswag_doc(doc: dict[str, Any], index: int) -> Instance | None:
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = _preprocess(doc["activity_label"] + ": " + ctx)
    choices = tuple(_preprocess(ending) for ending in doc["endings"])
    label = int(doc["label"]) if doc.get("label", "") != "" else -1

    return Instance(
        question=query,
        choices=choices,
        gold_answer=str(label),
        metadata={
            "id": doc.get("ind", index),
            "index": index,
            "dataset": "hellaswag",
            "gold_idx": label,
            "gold_text": choices[label] if 0 <= label < len(choices) else "",
        },
    )


def _format_rc(query: str, answer: str | None = None) -> str:
    if answer:
        return f"{query} {answer}"
    return query


def _format_mc(query: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"{query}\nChoose the best continuation:\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


@register("hellaswag")
class HellaSwag(Task):
    data_source = DataSource(path="allenai/hellaswag", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = self._load_hellaswag_instances()
        yield from self._instances_cache

    def _load_hellaswag_instances(self) -> list[Instance]:
        loader = DataLoader()
        instances: list[Instance] = []

        # When limit is set, load all splits (validation first, then train)
        # to match oe-eval-internal's split="all" ordering: test -> validation -> train.
        # HellaSwag has no test split, so: validation -> train.
        splits = ["validation", "train"] if self.config.limit else [self.config.split.value]

        index = 0
        for split in splits:
            source = self.config.get_data_source(split=split)
            for doc in loader.load(source):
                inst = self.process_doc(doc, index)
                if inst is not None:
                    instances.append(inst)
                    index += 1

        # Match oe-eval-internal's random subsampling: random.Random(1234).sample(docs, limit)
        if self.config.limit and len(instances) > self.config.limit:
            instances = random.Random(1234).sample(instances, self.config.limit)

        return instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return _process_hellaswag_doc(doc, index)

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_hellaswag_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in HELLASWAG_FIXED_FEWSHOT:
            inst = _process_hellaswag_doc(doc, 0)
            if inst is None:
                continue
            label = int(str(doc["label"])) if doc.get("label", "") != "" else -1
            choices = inst.choices or ()
            correct_ending = choices[label] if 0 <= label < len(choices) else ""
            letter = chr(ord("A") + label) if label >= 0 else ""
            instances.append(
                Instance(
                    question=inst.question,
                    choices=inst.choices,
                    gold_answer=correct_ending,
                    metadata={
                        "gold_idx": label,
                        "gold_text": correct_ending,
                        "mc_answer": letter,
                    },
                )
            )
        # Take the first num_fewshot examples (no random sampling) to match
        # oe-eval-internal's fewshot_source behavior which uses [:k].
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

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


register_variant("hellaswag", "rc")
register_variant("hellaswag", "mc", formatter=MultipleChoiceFormatter())
register_variant(
    "hellaswag",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_hellaswag_fixed",
)
register_variant(
    "hellaswag", "xlarge", num_fewshot=5, limit=10_000, fewshot_source="olmes_hellaswag_fixed"
)
register_variant("hellaswag", "bpb", metrics=(BPBMetric(),))
