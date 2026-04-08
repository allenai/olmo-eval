from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import ExactMatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.gsm_symbolic import GSM8K_FIXED_FEWSHOT
from olmo_eval.evals.tasks.gsm8k import _clean_short_answer, _extract_last_number


@register("gsm_symbolic")
class GSMSymbolic(Task):
    data_source = DataSource(path="apple/GSM-Symbolic", subset="main")
    metrics = (AccuracyMetric(scorer=ExactMatchScorer),)
    num_fewshot = 8
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0,
        stop_sequences=("Question:", "</s>", "<|im_end|>", "\n\n"),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc["question"]
        answer = doc["answer"]
        short_answer = answer.split("####")[-1].strip()
        cleaned = _clean_short_answer(short_answer)

        return Instance(
            question=question,
            gold_answer=cleaned,
            metadata={
                "id": index,
                "answer": answer,
                "short_answer": short_answer,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        instances = []
        for doc in GSM8K_FIXED_FEWSHOT:
            instances.append(
                Instance(
                    question=doc["question"],
                    gold_answer=doc["short_answer"],
                    metadata={
                        "answer": doc["answer"],
                        "short_answer": doc["short_answer"],
                    },
                )
            )
        num = self.config.num_fewshot
        if num and num < len(instances):
            instances = instances[:num]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()

        parts: list[str] = []
        for ex in fewshot:
            parts.append(f"Question: {ex.question}\nAnswer: {ex.metadata['answer']}")
        parts.append(f"Question: {instance.question}\nAnswer:")
        prompt = "\n\n".join(parts)

        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_last_number(output.text)


register_variant(
    "gsm_symbolic",
    "p1",
    data_source=DataSource(path="apple/GSM-Symbolic", subset="p1"),
)

register_variant(
    "gsm_symbolic",
    "p2",
    data_source=DataSource(path="apple/GSM-Symbolic", subset="p2"),
)

register_variant(
    "gsm_symbolic",
    "olmo3base",
    metrics=(
        AccuracyMetric(scorer=ExactMatchScorer),
        PassAtKMetric(k=1, scorer=ExactMatchScorer),
        PassAtKMetric(k=2, scorer=ExactMatchScorer),
        PassAtKMetric(k=4, scorer=ExactMatchScorer),
        PassAtKMetric(k=8, scorer=ExactMatchScorer),
    ),
    primary_metric=PassAtKMetric(k=1, scorer=ExactMatchScorer),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        stop_sequences=("Question:", "</s>", "<|im_end|>", "\n\n"),
        num_samples=8,
    ),
)
