from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant


@register("coqa:mc")
class CoqaMC(Task):
    data_source = DataSource(path="allenai/coqa_mc", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        query_original = doc.get("query_original", "")
        question = doc.get("question_original", "")
        if not question:
            return None

        choices_data = doc.get("choices", {})
        choices = choices_data.get("text", [])
        if not choices:
            return None

        answer_key = doc.get("answerKey", "")
        gold_idx = ord(answer_key) - ord("A") if answer_key else 0
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        passage_context = query_original.rsplit("Question:", 1)[0]

        return Instance(
            question=question,
            choices=tuple(choices),
            gold_answer=answer_key,
            metadata={
                "id": doc.get("id", f"coqa_mc_{index}"),
                "index": index,
                "dataset": "coqa_mc",
                "gold_idx": gold_idx,
                "gold_text": gold_text,
                "passage_context": passage_context,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        passage_context = instance.metadata.get("passage_context", "")
        choices = instance.choices or ()
        choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{passage_context}Question: {instance.question}\n{choices_text}\nAnswer:"
        continuations = tuple(f" {chr(ord('A') + i)}" for i in range(len(choices)))

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


register_variant("coqa:mc", "olmo3base")
