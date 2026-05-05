from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric, F1Metric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.naturalqs import NATURALQS_FIXED_FEWSHOT
from olmo_eval.evals.tasks.drop import DROPExactMatchScorer, DROPF1Scorer


def _format_query(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _normalize_answers(answers: Sequence[Any]) -> list[str]:
    return [str(answer) for answer in answers if str(answer).strip()]


@register("naturalqs")
class NaturalQs(Task):
    data_source = DataSource(path="google-research-datasets/nq_open", split="validation")
    split = Split.VALIDATION
    metrics = (
        F1Metric(scorer=DROPF1Scorer),
        AccuracyMetric(scorer=DROPExactMatchScorer),
    )
    primary_metric = F1Metric(scorer=DROPF1Scorer)
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
        stop_sequences=("Question:", "Q:", "\n\n"),
    )
    num_fewshot = 5
    fewshot_split = "train"
    _fewshot_source_name = "naturalqs_fixed"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question", "")).strip()
        answers = _normalize_answers(doc.get("answer", []))
        if not question or not answers:
            return None

        return Instance(
            question=_format_query(question),
            gold_answer=answers[0],
            metadata={
                "id": doc.get("index", index),
                "index": index,
                "all_answers": answers,
                "answers": [(answer,) for answer in answers],
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if getattr(self.config, "fewshot_source", None) == self._fewshot_source_name:
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for index, doc in enumerate(NATURALQS_FIXED_FEWSHOT):
            answers = _normalize_answers(doc["answer"])
            instances.append(
                Instance(
                    question=_format_query(str(doc["question"])),
                    gold_answer=", ".join(answers),
                    metadata={
                        "id": f"naturalqs_fixed_{index}",
                        "all_answers": answers,
                        "answers": [(answer,) for answer in answers],
                    },
                )
            )

        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        parts: list[str] = []
        for ex in self.get_fewshot():
            parts.append(f"{ex.question} {ex.gold_answer}")
        parts.append(instance.question)
        prompt = "\n\n".join(parts)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


register_variant("naturalqs", "gen")

register_variant(
    "naturalqs",
    "olmo3base",
    limit=10_000,
    seed=1234,
    fewshot_source="naturalqs_fixed",
)
