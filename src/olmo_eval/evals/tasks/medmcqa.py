from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.format_helpers import (
    format_mc as _format_mc,
)
from olmo_eval.evals.tasks.common.format_helpers import (
    format_rc as _format_rc,
)


@register("medmcqa")
class MedMCQA(Task):
    data_source = DataSource(path="openlifescienceai/medmcqa", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        choices = [
            doc.get("opa", ""),
            doc.get("opb", ""),
            doc.get("opc", ""),
            doc.get("opd", ""),
        ]

        gold_idx = int(doc.get("cop", 0))

        gold_text = choices[gold_idx]

        return Instance(
            question=question,
            choices=tuple(choices),
            gold_answer=chr(ord("A") + gold_idx),
            metadata={
                "id": doc.get("id", f"medmcqa_{index}"),
                "index": index,
                "dataset": "medmcqa",
                "gold_idx": gold_idx,
                "gold_text": gold_text,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.gold_answer or ""
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
                answer = ex.metadata.get("gold_text", ex.gold_answer or "")
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


register_variant("medmcqa", "rc")
register_variant("medmcqa", "mc", formatter=MultipleChoiceFormatter())
register_variant("medmcqa", "bpb", metrics=(BPBMetric(),))
register_variant("medmcqa", "olmo3base", num_fewshot=5, fewshot_seed=1234, seed=1234)
