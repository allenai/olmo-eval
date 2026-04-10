from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.piqa import PIQA_FIXED_FEWSHOT


@register("piqa")
class PiQA(Task):
    data_source = DataSource(path="piqa", split="validation", revision="refs/convert/parquet")
    split = Split.VALIDATION
    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        goal = doc.get("goal", "")
        sol1 = doc.get("sol1", "")
        sol2 = doc.get("sol2", "")
        label = int(doc.get("label", 0))

        if not goal:
            return None

        choices = (sol1, sol2)
        gold_text = choices[label] if 0 <= label < len(choices) else ""

        return Instance(
            question=goal,
            choices=choices,
            gold_answer=str(label),
            metadata={
                "id": index,
                "index": index,
                "dataset": "piqa",
                "gold_idx": label,
                "gold_text": gold_text,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_piqa_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in PIQA_FIXED_FEWSHOT:
            label = int(doc["label"])
            correct_sol = str(doc["sol1"] if label == 0 else doc["sol2"])
            letter = chr(ord("A") + label)
            instances.append(
                Instance(
                    question=str(doc["goal"]),
                    choices=(str(doc["sol1"]), str(doc["sol2"])),
                    gold_answer=correct_sol,
                    metadata={
                        "gold_idx": label,
                        "gold_text": correct_sol,
                        "mc_answer": letter,
                    },
                )
            )
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


def _format_mc(goal: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"Goal: {goal}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _format_rc(goal: str, answer: str | None = None) -> str:
    prompt = f"Goal: {goal}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


register_variant("piqa", "rc")

register_variant(
    "piqa",
    "mc",
    formatter=MultipleChoiceFormatter(),
)

register_variant(
    "piqa",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_piqa_fixed",
)

register_variant(
    "piqa",
    "olmes",
    num_fewshot=5,
    limit=1000,
    fewshot_source="olmes_piqa_fixed",
)

register_variant("piqa", "bpb", metrics=(BPBMetric(),))

register_variant("piqa", "full", limit=None)
