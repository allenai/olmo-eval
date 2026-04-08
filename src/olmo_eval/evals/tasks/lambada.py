from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, GreedyAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant


@register("lambada")
class LAMBADA(Task):
    data_source = DataSource(path="EleutherAI/lambada_openai")
    split = Split.TEST
    metrics = (GreedyAccuracyMetric(), BPBMetricByteAvg())
    primary_metric = GreedyAccuracyMetric()
    num_fewshot = 0
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        text = doc.get("text", "")
        if not text.strip():
            return None

        words = text.split()
        if len(words) < 2:
            return None

        answer_text = words[-1]
        query = " ".join(words[:-1])

        return Instance(
            question=query,
            gold_answer=answer_text,
            choices=(answer_text,),
            metadata={
                "id": index,
                "index": index,
                "gold_idx": 0,
                "gold_text": answer_text,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        continuation = " " + instance.gold_answer if instance.gold_answer else ""
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=(continuation,),
        )


register_variant("lambada", "bpb", formatter=PPLFormatter(), metrics=(BPBMetricByteAvg(),))

register_variant("lambada", "olmo3base")
