from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import SQuADF1Metric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.squad import SQUAD_FIXED_FEWSHOT

SQUAD_DESCRIPTION = "The following are reading comprehension questions, where the answer to each question is a segment of text from the corresponding background text."


def _format_query(title: str, context: str, question: str) -> str:
    return f"Title: {title}\nBackground: {context}\nQuestion: {question}\nAnswer:"


class SQuADBase(Task):
    fewshot_split = "train"
    metrics = (SQuADF1Metric(),)
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
        stop_sequences=("Title:", "\n\n"),
    )
    _fewshot_source_name = "squad_fixed"

    def _build_fewshot(self) -> list[Instance]:
        if getattr(self.config, "fewshot_source", None) == self._fewshot_source_name:
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in SQUAD_FIXED_FEWSHOT:
            answers = doc["answers"]
            assert isinstance(answers, dict)
            instances.append(
                Instance(
                    question=_format_query(
                        str(doc["title"]), str(doc["context"]), str(doc["question"])
                    ),
                    gold_answer=str(answers["text"][0]),
                    metadata={"id": str(doc["id"])},
                )
            )
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        title = doc.get("title", "")
        context = doc.get("context", "")
        question = doc.get("question", "")
        answers = doc.get("answers", {})
        answers_text = answers.get("text", [])
        if not question or not answers_text:
            return None
        return Instance(
            question=_format_query(title, context, question),
            gold_answer=answers_text[0],
            metadata={
                "id": doc.get("id", index),
                "index": index,
                "all_answers": answers_text,
            },
        )

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()

    def format_request(self, instance: Instance) -> LMRequest:
        parts: list[str] = [SQUAD_DESCRIPTION]
        for ex in self.get_fewshot():
            parts.append(f"{ex.question} {ex.gold_answer}")
        parts.append(instance.question)
        prompt = "\n\n".join(parts)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)


@register("squad")
class SQuAD(SQuADBase):
    data_source = DataSource(path="allenai/squad", split="validation")
    split = Split.VALIDATION
    num_fewshot = 5
    fewshot_source = "squad_fixed"


register_variant("squad", "gen")

register_variant(
    "squad",
    "olmo3base",
    limit=10_000,
    seed=1234,
    fewshot_source="squad_fixed",
)
