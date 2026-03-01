from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import F1Metric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant


@register("coqa")
class CoQA(Task):
    data_source = DataSource(path="EleutherAI/coqa")
    split = Split.VALIDATION
    metrics = (F1Metric(),)
    num_fewshot = 0
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
        stop_sequences=("\n\n",),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = self._load_coqa_instances()
        yield from self._instances_cache

    def _load_coqa_instances(self) -> list[Instance]:
        loader = DataLoader()
        instances: list[Instance] = []

        if self.config.limit:
            splits = ["train", "validation"]
        else:
            splits = [self.config.split.value]

        for split in splits:
            source = self.config.get_data_source(split=split)
            for doc in loader.load(source):
                instances.extend(self._process_doc_to_multi(doc))
                if self.config.limit and len(instances) >= self.config.limit:
                    break
            if self.config.limit and len(instances) >= self.config.limit:
                break

        if self.config.limit:
            instances = instances[: self.config.limit]

        return instances

    def _process_doc_to_multi(self, doc: dict[str, Any]) -> list[Instance]:
        story = doc["story"]
        questions = doc["questions"]["input_text"]
        all_answers = doc["answers"]["input_text"]
        source = doc.get("source", "")
        core_id = doc.get("id", "")
        additional_answers = [x["input_text"] for x in doc.get("additional_answers", {}).values()]

        instances: list[Instance] = []
        previous_qa: list[dict[str, str]] = []

        for turn_idx, question in enumerate(questions):
            answers = [all_answers[turn_idx]]
            for answer_list in additional_answers:
                if len(answer_list) > turn_idx and answer_list[turn_idx]:
                    answers.append(answer_list[turn_idx])

            query = f"Passage: {story}"
            if previous_qa:
                query += "\n\nPreceding questions:"
                for prev in previous_qa:
                    query += f"\n\nQuestion: {prev['question']}\nAnswer: {prev['answer']}"
            query += "\n\nFinal question:"
            query += f"\n\nQuestion: {question}\nAnswer:"

            instances.append(
                Instance(
                    question=query,
                    gold_answer=answers[0],
                    metadata={
                        "id": f"{core_id}_turn{turn_idx}",
                        "source": source,
                        "all_answers": answers,
                    },
                )
            )

            previous_qa.append({"question": question, "answer": answers[0]})

        return instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        multi = self._process_doc_to_multi(doc)
        return multi[0] if multi else None

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
        )


register_variant("coqa", "gen")

register_variant(
    "coqa",
    "olmo3base",
    limit=10_000,
    sampling_params=SamplingParams(
        max_tokens=50,
        temperature=0.0,
        stop_sequences=("\n\n", "Answer:", "Question:"),
    ),
)
