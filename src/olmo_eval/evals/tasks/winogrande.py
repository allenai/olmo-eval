from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.winogrande import WINOGRANDE_FIXED_FEWSHOT


def _partial_context(sentence: str, option: str) -> str:
    pronoun_loc = sentence.index("_")
    return sentence[:pronoun_loc] + option


def _partial_target(sentence: str) -> str:
    pronoun_loc = sentence.index("_") + 1
    return " " + sentence[pronoun_loc:].strip()


def _process_winogrande_doc(doc: dict[str, Any], index: int) -> Instance | None:
    sentence = doc.get("sentence", "")
    option1 = doc.get("option1", "")
    option2 = doc.get("option2", "")
    answer = doc.get("answer", "")

    if not sentence or "_" not in sentence:
        return None

    label = int(answer) - 1 if answer != "" else -1

    return Instance(
        question=sentence,
        choices=(option1, option2),
        gold_answer=str(label),
        metadata={
            "id": index,
            "index": index,
            "dataset": "winogrande",
            "gold_idx": label,
            "gold_text": (option1 if label == 0 else option2) if label >= 0 else "",
        },
    )


def _format_rc(sentence: str, option: str | None = None) -> str:
    if option:
        return _partial_context(sentence, option) + _partial_target(sentence)
    return sentence[: sentence.index("_")]


def _format_mc(sentence: str, options: tuple[str, ...], answer: str | None = None) -> str:
    prompt = f"Fill in the blank: {sentence.replace('_', '___')}"
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(options))
    prompt = f"{prompt}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


@register("winogrande")
class Winogrande(Task):
    data_source = DataSource(path="winogrande", subset="winogrande_xl", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = self._load_winogrande_instances()
        yield from self._instances_cache

    def _load_winogrande_instances(self) -> list[Instance]:
        loader = DataLoader()
        instances: list[Instance] = []

        splits = ["validation", "train"] if self.config.limit else [self.config.split.value]

        index = 0
        for split in splits:
            source = self.config.get_data_source(split=split)
            for doc in loader.load(source):
                inst = self.process_doc(doc, index)
                if inst is not None:
                    instances.append(inst)
                    index += 1

        if self.config.limit and len(instances) > self.config.limit:
            instances = instances[: self.config.limit]

        return instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return _process_winogrande_doc(doc, index)

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_winogrande_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in WINOGRANDE_FIXED_FEWSHOT:
            inst = _process_winogrande_doc(doc, 0)
            if inst is None:
                continue
            label = int(doc["answer"]) - 1 if doc.get("answer", "") != "" else -1
            choices = inst.choices or ()
            correct_option = choices[label] if 0 <= label < len(choices) else ""
            letter = chr(ord("A") + label) if label >= 0 else ""
            instances.append(
                Instance(
                    question=inst.question,
                    choices=inst.choices,
                    gold_answer=correct_option,
                    metadata={
                        "gold_idx": label,
                        "gold_text": correct_option,
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
            prompt = "\n\n".join(parts)
            return LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt=prompt,
                continuations=continuations,
            )
        else:
            # RC format uses Trinh & Le (2018) partial evaluation:
            # each option is placed in the prompt context, and only the target
            # (text after the blank) is scored as the continuation.
            sentence = instance.question
            target = _partial_target(sentence)
            fewshot_prompt = "\n\n".join(parts) if parts else ""
            continuation_prompts = tuple(
                ("\n\n".join([*parts, _partial_context(sentence, option)]))
                if parts
                else _partial_context(sentence, option)
                for option in (instance.choices or ())
            )
            continuations = tuple(target for _ in (instance.choices or ()))
            return LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt=fewshot_prompt,
                continuations=continuations,
                continuation_prompts=continuation_prompts,
            )


register_variant("winogrande", "rc")
register_variant("winogrande", "mc", formatter=MultipleChoiceFormatter())
register_variant(
    "winogrande",
    "olmo3base",
    num_fewshot=5,
    limit=10_000,
    fewshot_source="olmes_winogrande_fixed",
)
register_variant(
    "winogrande",
    "xlarge",
    num_fewshot=5,
    limit=10_000,
    fewshot_source="olmes_winogrande_fixed",
)
register_variant("winogrande", "bpb", metrics=(BPBMetric(),))
