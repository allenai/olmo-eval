from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import BPBMetric, LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant


@register("sciriff_yesno")
class SciriffYesNo(Task):
    data_source = DataSource(path="allenai/sciriff-yesno", split="train")
    split = Split.TRAIN
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        context = doc.get("context", "")
        question = doc.get("question", "")
        answer = doc.get("answer", "")

        if not question:
            return None

        choices = ("Yes", "No")
        gold_idx = 0 if answer == "Yes" else 1

        return Instance(
            question=question,
            choices=choices,
            gold_answer=str(gold_idx),
            metadata={
                "id": doc.get("id", index),
                "index": index,
                "dataset": "sciriff_yesno",
                "source": context,
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        source = instance.metadata.get("source", "")

        parts: list[str] = []
        for ex in fewshot:
            ex_source = ex.metadata.get("source", "")
            answer = ex.metadata.get("gold_text", "")
            parts.append(_format_rc(ex_source, ex.question, answer))

        parts.append(_format_rc(source, instance.question))

        prompt = "\n\n".join(parts)
        continuations = tuple(f" {c}" for c in (instance.choices or ()))

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


def _format_rc(source: str, question: str, answer: str | None = None) -> str:
    prompt = f"{source}\nQuestion: {question}\nAnswer:".strip()
    if answer:
        prompt += f" {answer}"
    return prompt


register_variant("sciriff_yesno", "rc")

register_variant("sciriff_yesno", "bpb", metrics=(BPBMetric(),))

register_variant(
    "sciriff_yesno",
    "olmes",
    num_fewshot=5,
    fewshot_seed=1234,
)

register_variant(
    "sciriff_yesno",
    "olmo3base",
    num_fewshot=5,
    fewshot_seed=1234,
)
